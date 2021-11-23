import joplin from 'api';
const Sqlite3 = joplin.plugins.require('sqlite3').verbose();
const Fs = joplin.plugins.require('fs-extra');
const Path = require('path');

//const Tf = require('@tensorflow/tfjs');
import * as Tf from '@tensorflow/tfjs';
const Use = require('@tensorflow-models/universal-sentence-encoder');

Tf.enableProdMode(); // not sure the extent to which this helps
//Tf.ENV.set('WEBGL_NUM_MB_BEFORE_PAGING', 4000);
//console.log(Tf.memory())

//Tf.setBackend('cpu');

// partial todo list
// - optimize if necessary (don't unstack tensors, *Sync() to *(), fix all await/async/promises)
// - - save USE model to disk so it's not redownloaded every time
// - - recompute embedding (and ALL similirities if we can limit cpu/gpu and do in bg) via event queue
// - ought to use event api for tracking note creation/updates/deletion
// - clean things up
// - - probably some large refactors doable, now that I understand flow better
// - manually test some edge cases?
// - UI issue that offsets note editor and renderer when width is made smaller
//   (I've seen this in other plugins too)
// - publish plugin (how?)
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

function deleteEmbedding(db, noteID) {
    const stmt = db.prepare("DELETE FROM note_embeddings WHERE note_id = ?");
    stmt.run(noteID).finalize();
    console.info('deleted ' + noteID);
}

async function loadEmbeddings(db) {
    console.info('loading embeddings');
    //    let prom = null;
    let notes = new Map();
    let stmt = null;
    db.serialize(function() {
	db.run("CREATE TABLE IF NOT EXISTS note_embeddings (note_id TEXT PRIMARY KEY, embedding TEXT);");
	//, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);");

	console.info('table exists');
	
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

    // console.log('rows', rows);
    for (const row of rows) {
	notes.set(row['note_id'], {id: row['note_id'], embedding: row['embedding'].split(" ").map(x => parseFloat(x))});
    }
    
    //prom = new Promise(function (resolve, reject) {resolve(notes)});
    //    let notes = await prom;
    //console.log('loading notes', [...notes.entries()]);
    return notes;
    //db.close();
}

function saveEmbeddings(db, idSlice, embeddings) {
    //console.info('saving', idSlice, embeddings);
    db.serialize(async function() {
	let stmt = db.prepare("INSERT INTO note_embeddings (note_id, embedding) VALUES (?,?) ON CONFLICT(note_id) DO UPDATE SET embedding = excluded.embedding");

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
	
	console.info('to db', stmt, idSlice, embeddings);
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
    embedding: Array<number>;
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
    for (const note of allNotes) {
	noteMap.set(note.id, {id: note.id, title: note.title, parent_id: note.parent_id, body: note.body})
    }
    return noteMap;
}

// consider looking at how doc2vec impls this for optimization inspo
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
    //let i = 0;
    for (const [id, n] of notes.entries()) {
	//console.log(i, id, n);
	//i += 1;
	const tensor2: Tf.Tensor = Tf.tensor1d(n.embedding);
	const x = Tf.dot(tensor1, tensor2.transpose());
	const y = x.dataSync();
	const score = y[0]; // returned as single element list, hence [0]
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

    //values.print();
    //indices.print();

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
    //console.log('notes: ', notes);
    let docs = [];
    for (const n of notes) {
	//docs.push(n.title);
	docs.push(n.title + "\n" + n.body);
    }
    return docs;
}

async function getAllNoteEmbeddings(model, db, panel) {
    let progressHTML = '<center><i>Computing/loading embeddings</i></center>';
    await updateHTML(panel, progressHTML);
    
    const allNotes = await getAllNotes();
    const allNoteIDs = [...allNotes.keys()];

    progressHTML += `<br /><br />Total # notes: ${allNoteIDs.length}`;
    await updateHTML(panel, progressHTML);
    
    // try loading saved embeddings first
    // determine which notes don't yet have embeddings, compute and save those

    // split the remaining notes needing to be embedded from allNotes,
    //   based on what was loaded
    const savedEmbeddings = await loadEmbeddings(db);     // map of noteID to 512dim array
    const knownIDs = [...savedEmbeddings.keys()];
    console.info('savedEmbeddings:', savedEmbeddings);
    const unembeddedIDs = allNoteIDs.filter(id => !knownIDs.includes(id));
    let remainingNotes = new Map();
    for (const nid of unembeddedIDs) {
    	remainingNotes.set(nid, allNotes.get(nid));
    }

    // todo use event queue to handle this better
    // delete notes from DB that are no longer in joplin proper
    const deletedIDs = knownIDs.filter(id => !allNoteIDs.includes(id));
    console.info('notes to delete from db: ', deletedIDs);
    for (const nid of deletedIDs) {
	deleteEmbedding(db, nid);
	savedEmbeddings.delete(nid);
    }

    progressHTML += `<br />Saved # embeddings: ${knownIDs.length}`;
    progressHTML += `<br />Remaining # embeddings: ${unembeddedIDs.length}`;
    await updateHTML(panel, progressHTML);

    // process the remaining notes
    const remaining_documents = notes2docs(remainingNotes.values());
    console.info('creating embeddings');
    //const tensors = await model.embed(['test']);
    let embeddings = [];
    const batch_size = 100;
    const num_batches = Math.floor(remaining_documents.length/batch_size);
    const remaining = remaining_documents.length % batch_size;
    console.info('batches to run ', num_batches, ' ', remaining);

    progressHTML += `<br /><br />Batch Size: ${batch_size} notes`;
    progressHTML += `<br /># full batches: ${num_batches}`;
    progressHTML += `<br /># notes in final partial batch: ${remaining}`;
    await updateHTML(panel, progressHTML);
    
    async function embed_batch(db, idSlice, slice) {
	//const model = await Use.load();
	//Tf.engine().startScope();
	const tensors = await model.embed(slice);
	//console.log(tensors)

	// prob don't want to do this for optimization reasons?
	// (prob faster to compute simlarity all in one go, vs iteratively for each tensor)
	// or maybe we want to untensorize them asap and dispose the tensors?
	const tensors_array = Tf.unstack(tensors);
	//console.log(tensors_array);
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
	// todo try tf.profile to understand model issue
    }

    progressHTML += "<br />";
    for (let i = 0; i < num_batches; i++) {
	const slice = remaining_documents.slice(i*batch_size, (i+1)*batch_size);
	const idSlice = unembeddedIDs.slice(i*batch_size, (i+1)*batch_size);
	
	//console.log(i, slice);
	let startTime = new Date().getTime();
	const e = await embed_batch(db, idSlice, slice);
	let endTime = new Date().getTime();
	let execTime = (endTime - startTime)/1000;
	//console.log('e: ', e)
	embeddings = embeddings.concat(e);
	//console.log('done ', i);

	console.info('batch ' + i, Tf.memory(), Tf.engine(), Tf.env());

	progressHTML += `<br />Finished batch ${i+1} in ${execTime} seconds`;
	await updateHTML(panel, progressHTML);
    }
    if (remaining > 0) {
	const slice = remaining_documents.slice(num_batches*batch_size);
	const idSlice = unembeddedIDs.slice(num_batches*batch_size);
	//console.log(slice);
	const e = await embed_batch(db, idSlice, slice);
	embeddings = embeddings.concat(e);
	
	progressHTML += `<br />Finished final batch`;
	await updateHTML(panel, progressHTML);
    }
    //const tensors = await model.embed(remaining_documents);
    //console.log('created', num_batches, ' ', remaining);

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
    //console.log('all notes with embeddings:', allNotes);

    return allNotes;
}

// borrowed from backlinks plugin: https://github.com/ambrt/joplin-plugin-referencing-notes/blob/master/src/index.ts
function escapeTitleText(text: string) {
    return text.replace(/(\[|\])/g, '\\$1');
}

// always keep title+scroll in html
async function updateHTML(panel, html) {
    const titleHTML = '<h3>Semantically Similar Notes</h3>';
    
    // css overflow-y allows scrolling,
    //   needs height specified so we use 100% of viewport height
    //   todo this doesn't seem to work for the embedding computation text.
    //   so maybe vh is heigh of app window, not of webview panel?
    // todo: copy default joplin styling.
    //   (can this be programmatically deteremined?)
    const scrollStyleHTML = `
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
    `;
    
    await joplin.views.panels.setHtml(panel, titleHTML + scrollStyleHTML +
				             `<div class="scroll_enabled">` +
 				             html +
				             `</div>`);
}

joplin.plugins.register({
    onStart: async function() {
	await Tf.ready(); // any perf issue of keeping this in prod code?
	console.info('tensorflow backend: ', Tf.getBackend());
	//console.log(Tf.memory())


	const selectNotePromptHTML = '<br /><i><center>Select a note to see similar notes</center></i>'

	// Create the panel object
	const panel = await joplin.views.panels.create('semanticlly_similar_notes_panel');
	await joplin.views.panels.onMessage(panel, async (message) => {
	    await joplin.commands.execute("openNote", message.noteId)
	});

	const pluginDir = await joplin.plugins.dataDir();
	const embeddingsDBPath = Path.join(pluginDir, 'embeddings.sqlite');
	console.info('Checking if "' + pluginDir + '" exists:', await Fs.pathExists(pluginDir));

	const db = openDB(embeddingsDBPath);

	// the Favorites plugin does smt similar to what I envison wrt UI element
	// (ie, it looks like the main note list in joplin)
	//   https://emoji.discourse-cdn.com/twitter/house.png?v=10
	async function updateUIWithNoteList(similar_notes) {
	    const html_links = []
	    for (const n of similar_notes) {
		const ahref = `<i>(${n.relative_score}%)</i> <a href="#" onclick="webviewApi.postMessage({type:'openNote',noteId:'${n.id}'})">${escapeTitleText(n.title)}</a>`
		html_links.push(ahref);
	    }

	    await updateHTML(panel, `${html_links.join('<br /><br />')}`);
	}

	await updateHTML(panel, '<center><i>Downloading model from Tensorflow Hub</i></center>')
	const model = await Use.load();
	console.info(Tf.memory())
	console.info(model);
	
	// not sure what i'm doing with this async/await stuff...
	// think I ought to rethink the design around this
	// notes is map of id to note
	let notes = await getAllNoteEmbeddings(model, db, panel);
	// todo move part of this function inside updateSimilarNoteList
	//  so that new note title names are accurate. but don't want to relaod
	//  everything from DB

	await updateHTML(panel, selectNotePromptHTML);

	// if reEmbed,
	//   this will compute the embedding for the selected note,
	//   update the var in which we store all notes,
	//   and save the new embedding to the db.
	// regardless of reEmbed, this will:
	//   compute the similarities to all other notes,
	//   and display them in sorted order in the WebView
	// todo could conditionally recompute similarities, too
	async function updateSimilarNoteList(updateType: string, reEmbed: boolean) {
	    console.info('updating bc: ', updateType)
	    // Get the current note from the workspace.
	    const note = await joplin.workspace.selectedNote();

	    // Keep in mind that it can be `null` if nothing is currently selected!
	    if (note) {
		console.info('selected note title:\n', note.title);

		await updateHTML(panel, 'Computing similarities...');

		let embedding = null;
		let noteObj = notes.get(note.id);
		
		// if there is no note object, it's a new note, so create note obj
		// and "re"Embed it
		if (!noteObj) {
		    reEmbed = true;
		    noteObj = {id: note.id, title: note.title,
			       parent_id: note.parent_id, body: note.body,
			       embedding: null // will be set in a sec
			      }
		}
		    
		if (reEmbed) { 
		    const [document] = notes2docs([note]);
		    //console.info('document:\n', document);

		    const tensor = await model.embed(document);
		    // tensor is 512dim embedding of document
		
		    // update our embedding of this note
		    //console.log('pre tensing', tensor);
		    embedding = tensor.arraySync()[0];
		    noteObj['embedding'] = embedding;
		    notes.set(note.id, noteObj);
		    tensor.dispose(); // dispose here but create in search_similar_embeddings -> prob slow
		    
		    // persist the calculated embedding to disk
		    // todo anyway to detect if the change doesn't make it?
		    //  eg if pc lost power between the joplin note saving to disk
		    //  and this func saving the corresponding new embedding,
		    //  then results would be off until next time user edits this note
		    // - could compare timestamp of last note change with timestamp
		    //   of last embedding change on startup
		    //console.log('test before save');
		    saveEmbeddings(db, [note.id], [embedding]);
		} else {
		    embedding = noteObj['embedding'];
		}

		//console.log('tensing', embedding);
		const [sorted_note_ids, similar_note_scores] = search_similar_embeddings(embedding, notes);
		//console.log(sorted_note_ids, similar_note_scores);

		// todo optimize this...
		// - keep things as tensors?
		// - do large tensor multiplication of all note sims at once?
		//   could do for 1:N note sims, but maybe also N:N?
		let sorted_notes = [];
		for (let i = 0; i < notes.size; i++) {
		    //for (const nidx of sorted_note_ids) {
		    const nidx = sorted_note_ids[i];

		    // don't link to ourself (prob always index 0? hopefully...)
		    if (nidx == note.id) {
			continue;
		    }
		    
		    const n = notes.get(nidx);
		    n['relative_score'] = (similar_note_scores[i]*100).toLocaleString(undefined, {maximumSignificantDigits: 2});
		    sorted_notes.push(n);
		    //console.info(n.title, ": ", similar_note_scores[i]);
		}
		
		updateUIWithNoteList(sorted_notes);

		// webgl BE requires manual mem mgmt.
		// use tf.tidy to reduce risk of forgetting to call dispose

		//model.dispose();
	    } else {
		await updateHTML(panel, selectNotePromptHTML);
	    }
	}

	// TODO for snappier performance, I think we could to listen to the event queue
	// and recompute embeddings whenever any notes changes. then just run the
	// similarity comparison when the selected ntoe changes (or is updated).
	// could potentially recompute all similarities in the background too,
	// but that might be too much wasted computation to be worth it.
	// for now, we just recompute the embedding whenever the note
	// selection changes (or when selected note's content changes).
	// - event api: https://joplinapp.org/api/references/rest_api/
	// - ought to use event queue for deleting notes, too...
	
	// This event will be triggered when the user selects a different note
	await joplin.workspace.onNoteSelectionChange(() => {
	    updateSimilarNoteList('note selection', true);
	});

	// This event will be triggered when the content of the selected note changes
	await joplin.workspace.onNoteChange(() => {
	    // commenting this out bc its too distracting to refresh list
	    // every few keystrokes. let user just switch notes back n forth
	    // if they want to recompute...
	    // updateSimilarNoteList('note change', true);
	});

	// await joplin.settings.onChange( async () => {
	//     updateSimilarNoteList();
	// });

	updateSimilarNoteList('startup', false);
    },
});

