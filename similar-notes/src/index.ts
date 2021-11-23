import joplin from 'api';
const Sqlite3 = joplin.plugins.require('sqlite3').verbose();
const Fs = joplin.plugins.require('fs-extra');
const Path = require('path');

//const Tf = require('@tensorflow/tfjs');
import * as Tf from '@tensorflow/tfjs';
const Use = require('@tensorflow-models/universal-sentence-encoder');

Tf.enableProdMode(); // not sure the extent to which this helps
//Tf.ENV.set('WEBGL_NUM_MB_BEFORE_PAGING', 4000);
console.log(Tf.memory())

const MAX_DOCUMENTS = 1000; // just for debugging

//Tf.setBackend('cpu');

// - test with my 900 notes
// - - why does model die around 785-790 notes??
// - - - maybe USE docs too long? -> yes, only encoding titles lets me get through all notes w/o hang
// - - - - could split long notes and average their vectors? would that impact perf?
// - - - - i ahve 1 note that's 1mb, next biggest 100kb. testing w/o 1mb note -> yes it works!
// - - - - - so, maybe cut off each note at 200kb? and future work to avg 200kb chunks together
// - - - - - implies that, say, 10000 notes would crash model too :(
// - - - - - - actually maybe not, might just constrain size of single document
// - - results in memleak - closing joplin still leaves process hogging a bunch of cpu
// - - looks like number of gpu bytes allocatd keeps growing for some reason, every batch
// - - also hangs when i try calling model.dispose()
// - - trace when gpu_init gets called?
// - - consider switching to mobileBERT? but mb test in python first?
// - - workaround: save batche of embeddings to disk, and then just restart process as often as needed
//     to get through all notes. yeesh...
// - optimize if necessary (don't unstack tensors, *Sync() to *(), fix all await/async/promises)
// - - save USE model to disk so it's not redownloaded every time
// - - save/load embeddings so they needn't be recalc'd every time joplin opens
// - - recompute embedding (and ALL similirities if we can limit cpu/gpu and do in bg) in onNoteChange
// - critical todos (eg tensor dispose)
// - clean things up
// - UI issue that offsets note editor and renderer when width is made smaller
//   (I've seen this in other plugins too)
// - write README
// - publish plugin
// - compare semantic similarity results with full USE model, vs this USE lite model

function openDB(embeddingsDBPath) {
    let db = new Sqlite3.Database(embeddingsDBPath, (err) => {
	if (err) {
	    console.error(err.message);
	    // TODO what to do for main plugin logic? throw exception? return null?
	    //return null;
	    throw err;
	} else {
	    console.log('Connected to embeddings db at ', embeddingsDBPath);
	}
    });
    
    return db;
}


// async?
async function loadEmbeddings(db) {
    console.log('loading embeddings');
    //    let prom = null;
    let notes = new Map();
    let stmt = null;
    db.serialize(function() {
	db.run("CREATE TABLE IF NOT EXISTS note_embeddings (note_id TEXT PRIMARY KEY, embedding TEXT);");
	//, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);");

	console.log('table exists');
	
	stmt = db.prepare("SELECT note_id, embedding FROM note_embeddings");
    });

    // sqlite3 doesn't use await/async, so we make our own
    const rows: Array<object> = await new Promise((resolve, reject) => {
	stmt.all(function(err, rows) {
	    if (err) { reject(err); }
	    resolve(rows);
	});
	stmt.finalize();
    }); // todo throw error on reject

    console.log('rows', rows);
    for (const row of rows) {
	notes.set(row['note_id'], {id: row['note_id'], embedding: row['embedding'].split(" ").map(x => parseFloat(x))});
    }
    
    //prom = new Promise(function (resolve, reject) {resolve(notes)});
    //    let notes = await prom;
    console.log('loading notes', [...notes.entries()]);
    return notes;
    //db.close();
}

function saveEmbeddings(db, idSlice, embeddings) {
    console.log('saving', idSlice, embeddings);
    db.serialize(async function() {
	let stmt = db.prepare("INSERT INTO note_embeddings (note_id, embedding) VALUES (?,?)");

	// this promise isn't doing what i want. want to essentially force db commit to happen
	// bc otherwise model crashes the program before things get written... TODO
	await new Promise((resolve, reject) => {
	    for (var i = 0; i < idSlice.length; i++) {
		//console.log(idSlice[i].toString(), ' and ', embeddings[i].join(" "));
		stmt.run(idSlice[i].toString(), embeddings[i].join(" "));
	    }

	    stmt.finalize();
	    resolve();
	});
	
	console.log('to db', stmt);
    });
}


// async function loadEmbeddings() {
//     try {
// 	// check if file exists. if not, assume this is the first time user is using this plugin
// 	// todo
	
// 	Embeddings = await fs.readJson(EmbeddingsJSONPath)
// 	console.log('loaded embeddings');
//     } catch (err) {
// 	console.error(err)
//     }
// }

// async function saveEmbeddings() {
//     try {
// 	await fs.writeJson(EmbeddingsJSONPath, Embeddings)
// 	console.log('saved embeddings')
//     } catch (err) {
// 	console.error(err)
//     }
// }

// async function loadModel() {
//     // if we already have it saved from disk, load from there
// python ref, but mb helpful: https://stackoverflow.com/questions/69949405/save-and-load-universal-sentence-encoder-model-on-different-machines
//     // otherwise, download from tfhub and save it to disk
// }

interface Note {
    id: string;
    parent_id: string;
    title: string;
    body: string;
    embedding: Tf.Tensor;
    // we also shim in a score attribute...
}

// code borrowed from joplin link graph plugin
async function getAllNotes(): Promise<Map<string, Note>> {
    var allNotes = []
    var page_num = 1;
    do {
	// `parent_id` is the ID of the notebook containing the note.
	var notes = await joplin.data.get(['notes'], {
	    fields: ['id', 'parent_id', 'title', 'body'],
	    // for semantic similarity, updated_time seems like an irrelevant ordering.
	    // maybe, extract top keywords from current note, FTS those, order by...relevancy?
	    // - just a potential optimization to display semantically similar notes faster.
	    //   not actually sure how long takes to getAllNotes in practice.
	    order_by: 'updated_time',
	    order_dir: 'DESC',
	    limit: 100,
	    page: page_num,
	});
	allNotes.push(...notes.items);
	page_num++;
    } while (notes.has_more)

    const noteMap = new Map();
    let i = 0;
    for (const note of allNotes) {
	i = i +1;
	if (i > MAX_DOCUMENTS) { // just for testing TODO
	    break;
	}
	noteMap.set(note.id, {id: note.id, title: note.title, parent_id: note.parent_id, body: note.body})
    }
    return noteMap;
}

// TODO look at how doc2vec impls this
function search_similar_embeddings(embedding, notes) {
    // tensor is 1x512
    // tensors is Nx512 where N = # notes
    
    // top2vec's impl in python for 1 to many string similarity search
    //ranks = np.inner(vectors, vector)
    //indexes = np.flip(np.argsort(ranks)[-num_res:])
    //scores = np.array([ranks[res] for res in indexes])
    //return indexes, scores

    // this is equiv of np.inner
    // todo why does official tf USE readme not use Tf.dot?
    let scores = [];
    let ids = [];
    //const num_tensors = tensors.arraySync()[0].length
    //Tf.unstack(tensors).forEach(t => t.print(true));
    // todo extend tensor to same dim as tensors, and do mult in 1 op, vs forEach
    //const flipped = Tf.transpose(tensor);
    //Tf.unstack(tensors).forEach(t => scores.push(Tf.dot(tensor, t)));

    //console.log(ts.length)
    //console.log(notes);
    //console.log(embedding); // this prints a 512dim even after gpu_init error
    const tensor1 = Tf.tensor1d(embedding);
    for (const [id, n] of notes.entries()) {
	//console.log(id, n);
	const tensor2: Tf.Tensor = Tf.tensor1d(n.embedding);
	const x = Tf.dot(tensor1, tensor2.transpose());
	const y = x.dataSync();
	const score = y[0]; // TODO why [0]? //parseFloat(y);
	//console.log(score);

	tensor2.dispose();
	x.dispose();
	//tensor.print(true);
	//t.print(true);
	//score.print(true);
	//console.log(score.dataSync()); // not a tensor, just an array32Float
	//console.log(parseFloat(score.dataSync())); // normal js float
	ids.push(id);
	scores.push(score);
    }
    tensor1.dispose();
    // for (let i = 0; i < num_tensors; i++) {
    // 	console.info('dotting ', tensor, ' and ', tensors[i]);
    // 	scores.push(Tf.dot(tensor, tensors[i]));
    // 	//scores.push(dotProduct(tensor, tensors[i]));
    // }
    // for (const t of scores) {
    // 	t.print();
    // }
    
    const {values, indices} = Tf.topk(scores, scores.length);
//    const syncedIndices: Array<number> = Array.from(indices.arraySync());
    // console.log(indices);
    // console.log(syncedIndices);
    // for (const i of indices) {
    //  	i.print();
    // }

    values.print();
    indices.print();

    //    const ia: Array<number> = Array.from([indices.arraySync()]);
    const ia = indices.arraySync();
    const syncdValues = values.arraySync();
    
    let sorted_note_ids: Array<number> = [];
    for (let i = 0; i < notes.size; i++) {
	const id_index = ia[i];
	sorted_note_ids.push(ids[id_index]);
    }
    

    values.dispose();
    indices.dispose();
    
    return [sorted_note_ids, syncdValues];
    

}
function notes2docs(notes) {
    console.log('notes: ', notes);
    let docs = [];
    for (const n of notes) {
	//docs.push(n.title);
	docs.push(n.title + "\n" + n.body);
    }
    return docs;
}

async function getAllNoteEmbeddings(model, db) {
    const allNotes = await getAllNotes();
    const allNoteIDs = [...allNotes.keys()];

    // try loading saved embeddings first
    // determine which notes don't yet have embeddings, compute and save those

    // split the remaining notes needing to be embedded from allNotes,
    //   based on what was loaded
    const savedEmbeddings = await loadEmbeddings(db);     // map of noteID to 512dim array
    const knownIDs = [...savedEmbeddings.keys()];
    console.log('savedEmbeddings:', savedEmbeddings);
    const unembeddedIDs = allNoteIDs.filter(id => !knownIDs.includes(id));
    let remainingNotes = new Map();
    for (const nid of unembeddedIDs) {
    	remainingNotes.set(nid, allNotes.get(nid));
    }

    // process the remaining notes
    const remaining_documents = notes2docs(remainingNotes.values());
    console.log('creating embeddings');
    //const tensors = await model.embed(['test']);
    let embeddings = [];
    const batch_size = 100;
    const num_batches = Math.floor(remaining_documents.length/batch_size);
    const remaining = remaining_documents.length % batch_size;
    console.log(num_batches, ' ', remaining);

    async function embed_batch(db, idSlice, slice) {
	//const model = await Use.load();
	//Tf.engine().startScope();
	const tensors = await model.embed(slice);
	console.log(tensors)

	// prob don't want to do this for optimization reasons?
	// (prob faster to compute simlarity all in one go, vs iteratively for each tensor)
	// or maybe we want to untensorize them asap and dispose the tensors?
	const tensors_array = Tf.unstack(tensors);
	console.log(tensors_array);
	let embeddings = [];
	for (const t of tensors_array) {
	    const a = t.arraySync(); // TODO why doesn't this need [0] but other arraySyncs do?
	    //console.log(t, a);
	    embeddings.push(a); 
	    t.dispose();
	}
	tensors.dispose();
	//Tf.disposeVariables(); // don't think we use any vars but just trying things
	//Tf.engine().endScope();
	//model.dispose(); //this causes things to hang for some reason
	//	model.reset_default_graph();
	//model.layers.forEach(l => l.dispose());

	// originally designed this way to accommodate model crashing on large input, 
	// but didn't end up figuring out how to force commit to DB before moving on,
	// so ought to be refactored...
	saveEmbeddings(db, idSlice, embeddings);
	
	return embeddings;
	// TODO try tf.profile
	// TODO try sleeping between batches? lol idk
    }

    for (let i = 0; i < num_batches; i++) {
	const slice = remaining_documents.slice(i*batch_size, (i+1)*batch_size);
	const idSlice = unembeddedIDs.slice(i*batch_size, (i+1)*batch_size);
	
	//console.log(i, slice);
	const e = await embed_batch(db, idSlice, slice);
	//console.log('e: ', e)
	embeddings = embeddings.concat(e);
	console.log('done ', i);

	console.log(Tf.memory(), Tf.engine(), Tf.env());
    }
    if (remaining > 0) {
	const slice = remaining_documents.slice(num_batches*batch_size);
	const idSlice = unembeddedIDs.slice(num_batches*batch_size);
	//console.log(slice);
	const e = await embed_batch(db, idSlice, slice);
	embeddings = embeddings.concat();
    }
    //const tensors = await model.embed(remaining_documents);
    console.log('created', num_batches, ' ', remaining);

    // create full Note objects based on loaded embeddings and created embeddings
    // savedEmbeddings has id->{embedding} of loaded embeddings
    for (const [nid, note] of savedEmbeddings) {
	const n = allNotes.get(nid);
	n['embedding'] = note.embedding;
	allNotes.set(nid, n);
    }
    
    //console.log(embeddings);
    // const keys = [....keys()];
    //    for (const nid of unembeddedIDs) {
    
    // embeddings is array of created embeddings
    // unembeddedIDs is array of noteIDs, same order+length as embeddings
    for (let i = 0; i < unembeddedIDs.length; i += 1) {
	const nid = unembeddedIDs[i];
	let n = remainingNotes.get(nid);
	n['embedding'] = embeddings[i];
	//console.log(n);
	allNotes.set(nid, n);
	//embedding_map[allNotes[i].id] = tensors_array[i];
    }
    console.log('all notes with embeddings:', allNotes);

    // TODO need to dispose of tensors at some point. is there a joplin onShutdown?

    return allNotes;
}

// borrowed from backlinks plugin: https://github.com/ambrt/joplin-plugin-referencing-notes/blob/master/src/index.ts
function escapeTitleText(text: string) {
    return text.replace(/(\[|\])/g, '\\$1');
}

joplin.plugins.register({
    onStart: async function() {
	await Tf.ready(); // any perf issue of keeping this in prod code?
	console.info('tensorflow backend: ', Tf.getBackend());
	//console.log(Tf.memory())
	
	// Create the panel object
	const panel = await joplin.views.panels.create('semanticlly_similar_notes_panel');
	await joplin.views.panels.setHtml(panel, '<br /><i><center>Select a note to see similar notes</center></i>');
	await joplin.views.panels.onMessage(panel, async (message) => {
	    await joplin.commands.execute("openNote", message.noteId)
	});

	const pluginDir = await joplin.plugins.dataDir();
	const embeddingsDBPath = Path.join(pluginDir, 'embeddings.sqlite');
	console.info('Checking if "' + pluginDir + '" exists:', await Fs.pathExists(pluginDir));

	const db = openDB(embeddingsDBPath);

	// todo see toc plugin for basics of adding/styling/interactivizing FE UI element
	//   https://joplinapp.org/api/tutorials/toc_plugin/
	//
	// also the Favorites plugin does smt similar to what I envison wrt UI element
	//   https://emoji.discourse-cdn.com/twitter/house.png?v=10
	async function updateWebView(similar_notes) {
	    const html_links = []
	    for (const n of similar_notes) {
		const ahref = `(${n.relative_score}%) <a href="#" onclick="webviewApi.postMessage({type:'openNote',noteId:'${n.id}'})">${escapeTitleText(n.title)}</a>`
		html_links.push(ahref);
	    }

	    // css overflow-y allows scrolling, needs height specified so we use 100% of viewport height
	    // todo: copy default joplin styling. (can this be programmatically deteremined?)
	    await joplin.views.panels.setHtml(panel, `
					<style>
					.scroll_enabled {
					    overflow-y: auto;
					    max-height: 100vh;
					}
					.scroll_enabled::-webkit-scrollbar {
					    width: 15px;
					}
					.scroll_enabled::-webkit-scrollbar-corner {
					    background: rgba(0,0,0,0);
					}
					.scroll_enabled::-webkit-scrollbar-thumb {
					    background-color: #ccc;
					    border-radius: 6px;
					    border: 4px solid rgba(0,0,0,0);
					    background-clip: content-box;
					    min-width: 32px;
					    min-height: 32px;
					}
					.scroll_enabled::-webkit-scrollbar-track {
					    background-color: rgba(0,0,0,0);
					}
					</style>
					<h3>Semantically Similar Notes</h3>
					<div class="scroll_enabled">
						${html_links.join('<br /><br />')}
					</div>
				`);
	}

	const model = await Use.load();
	console.log(Tf.memory())
	console.log(model);
	
	// TODO not sure what i'm doing with this async/await stuff
	// think I ought to rethink thru the design around this
	const notes = await getAllNoteEmbeddings(model, db);
	
	// todo await?
	// loadEmbeddings();

	// this will modify the global Embeddings variable for the given note,
	// compute the new similarities to all other notes,
	// and display them in sorted order in the WebView
	async function updateSimilarNoteList(updateType: string) {
	    console.info('updating bc: ', updateType)
	    // Get the current note from the workspace.
	    const note = await joplin.workspace.selectedNote();

	    // Keep in mind that it can be `null` if nothing is currently selected!
	    if (note) {
		console.info('selected note title:\n', note.title);

		await joplin.views.panels.setHtml(panel, 'Computing similarities...');

		const [document] = notes2docs([note]);
		//console.info('document:\n', document);

		model.embed(document).then(tensor => { // tensor is 512dim embedding of document
		    // update our embedding of this note
		    //console.log('pre tensing', tensor);
		    const embedding = tensor.arraySync()[0];
		    const n = notes.get(note.id);
		    n['embedding'] = embedding; // update embedding (todo: move to onNoteChange)
		    notes.set(note.id, n);
		    tensor.dispose(); // dispose here but create in search_similar_embeddings -> prob slow

		    //console.log('tensing', embedding);
		    const [sorted_note_ids, similar_note_scores] = search_similar_embeddings(embedding, notes);
		    //console.log(sorted_note_ids, similar_note_scores);

		    // todo optimize this...
		    let sorted_notes = [];
		    for (let i = 0; i < notes.size; i++) {
			//for (const nidx of sorted_note_ids) {
			const nidx = sorted_note_ids[i];

			// don't link to ourself (prob always index 0? hopefully...)
			if (nidx == note.id) {
			    continue;
			}
			
			const n: Note = notes.get(nidx);
			n['relative_score'] = (similar_note_scores[i]*100).toLocaleString(undefined, {maximumSignificantDigits: 2});
			sorted_notes.push(n);
		        //console.info(n.title, ": ", similar_note_scores[i]);
		    }
		    
		    updateWebView(sorted_notes);

		    // webgl BE requires manual mem mgmt.
		    // todo use tf.tidy to reduce risk of forgetting to call dispose

		    // TODO
		    //tensor.dispose();
		});
		
		//model.dispose();
	    } else {
		await joplin.views.panels.setHtml(panel, '<br />Select a note to see a list of semantically similar notes');
	    }
	}

	// This event will be triggered when the user selects a different note
	await joplin.workspace.onNoteSelectionChange(() => {
	    updateSimilarNoteList('note selection');
	});

	// This event will be triggered when the content of the note changes
	await joplin.workspace.onNoteChange(() => {
	    // this will update global Embeddings for us, compare to other notes, and show user results
	    updateSimilarNoteList('note change');

	    // TODO need to save this

	    // so we just need to store the updated embedding in the filesystem
	    // saveEmbeddings();

	});

	// await joplin.settings.onChange( async () => {
	//     updateSimilarNoteList();
	// });

	//updateSimilarNoteList('startup');
    },
});

