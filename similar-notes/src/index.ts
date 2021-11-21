import joplin from 'api';
//const Fs = joplin.plugins.require('fs-extra');
//const Path = require('path');

//const Tf = require('@tensorflow/tfjs');
import * as Tf from '@tensorflow/tfjs';
const Use = require('@tensorflow-models/universal-sentence-encoder');

// - improve webview: scrollview
// - test with my 800 notes
// - optimize if necessary (save/load embeddings, don't unstack tensors, *Sync() to *(), fix all await/async/promises)
// - fix cross platform issues
// - critical todos (eg tensor dispose)
// - clean things up
// - write README
// - publish plugin

// const PluginDir = await joplin.plugins.dataDir();
// const EmbeddingsJSONPath = Path.join(PluginDir, 'embeddings.json');
// console.info('Checking if "' + PluginDir + '" exists:', await fs.pathExists(PluginDir));

//var All_notes: Map<string, Note> = {}

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

interface Note {
    id: string;
    parent_id: string;
    title: string;
    body: string;
    embedding: Tf.Tensor;
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

// TODO look at how doc2vec impls this
function search_similar_embeddings(tensor, notes) {
    // tensor is 1x512
    // tensors is Nx512 where N = # notes
    
    // top2vec's impl in python for 1 to many string similarity search
    //ranks = np.inner(vectors, vector)
    //indexes = np.flip(np.argsort(ranks)[-num_res:])
    //scores = np.array([ranks[res] for res in indexes])
    //return indexes, scores

    // this is equiv of np.inner
    // todo why does official tf USE readme not use Tf.dot?
    var scores = [];
    var ids = [];
    //const num_tensors = tensors.arraySync()[0].length
    //Tf.unstack(tensors).forEach(t => t.print(true));
    // todo extend tensor to same dim as tensors, and do mult in 1 op, vs forEach
    //const flipped = Tf.transpose(tensor);
    //Tf.unstack(tensors).forEach(t => scores.push(Tf.dot(tensor, t)));

    //console.log(ts.length)
    //console.log(notes);
    for (const [id, n] of notes.entries()) {
	console.log(id, n);
	const tensor2: Tf.Tensor = n.embedding;
	const x = Tf.dot(tensor, tensor2.transpose());
	const y = x.dataSync();
	const score = y[0]; // TODO why [0]? //parseFloat(y);
	console.log(score);
	//tensor.print(true);
	//t.print(true);
	//score.print(true);
	//console.log(score.dataSync()); // not a tensor, just an array32Float
	//console.log(parseFloat(score.dataSync())); // normal js float
	ids.push(id);
	scores.push(score);
    }
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
    
    var sorted_note_ids: Array<number> = [];
    for (var i = 0; i < notes.size; i++) {
	const id_index = ia[i];
	sorted_note_ids.push(ids[id_index]);
    }
    
    // sort the scores, but keep track of the index they were in? aka np.flip + np.argsort I think
    // then retrieve scores
    
    return [sorted_note_ids, values.arraySync()];
    
    // for tensors to tensors: (todo replace with np.inner, still)
    // var scores = [];
    // for (let i = 0; i < tensor1.length; i++) {
    // 	for (let j = 0; j < tensor2.length; j++) {
    // 	    scores.push(dotProduct(tensor1[i], tensor2[j]));
    // 	}
    // }

}

// // Calculate the dot product of two vector arrays.
// const dotProduct = (xs, ys) => {
//     const sum = xs => xs ? xs.reduce((a, b) => a + b, 0) : undefined;

//     return xs.length === ys.length ?
// 	sum(zipWith((a, b) => a * b, xs, ys))
// 	: undefined;
// }

// // zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
// const zipWith =
//       (f, xs, ys) => {
// 	  const ny = ys.length;
// 	  return (xs.length <= ny ? xs : xs.slice(0, ny))
// 	      .map((x, i) => f(x, ys[i]));
//       }

function notes2docs(notes) {
    console.log('notes: ', notes);
    var docs = [];
    for (const n of notes) {
	docs.push(n.title + "\n" + n.body);
    }
    return docs;
}

async function getAllNoteEmbeddings() {
    const all_notes = await getAllNotes();
    const all_documents = notes2docs(all_notes.values());
    const tensors = await Use.load().then(model => model.embed(all_documents));

    // prob don't want to do this for optimization reasons?
    // (prob faster to compute simlarity all in one go, vs iteratively for each tensor)
    const tensors_array = Tf.unstack(tensors);

    //embedding_map = {}
    const keys = [...all_notes.keys()];
    for (var i = 0; i < all_notes.size; i += 1) {
	const nid = keys[i];
	var n = all_notes.get(nid);
	n['embedding'] = tensors_array[i];
	all_notes.set(nid, n);
	//embedding_map[all_notes[i].id] = tensors_array[i];
    }

    // TODO need to dispose of tensors at some point. is there a joplin onShutdown?

    return all_notes;
}

// borrowed from backlinks plugin: https://github.com/ambrt/joplin-plugin-referencing-notes/blob/master/src/index.ts
function escapeTitleText(text: string) {
    return text.replace(/(\[|\])/g, '\\$1');
}

joplin.plugins.register({
    onStart: async function() {
	await Tf.ready(); // any perf issue of keeping this in prod code?
	console.info('tensorflow backend: ', Tf.getBackend());

	// Create the panel object
	const panel = await joplin.views.panels.create('semanticlly_similar_notes_panel');
	await joplin.views.panels.setHtml(panel, 'Select a note to see similar notes');
	await joplin.views.panels.onMessage(panel, async (message) => {
	    await joplin.commands.execute("openNote", message.noteId)
	});


	// todo see toc plugin for basics of adding/styling/interactivizing FE UI element
	//   https://joplinapp.org/api/tutorials/toc_plugin/
	//
	// also the Favorites plugin does smt similar to what I envison wrt UI element
	//   https://emoji.discourse-cdn.com/twitter/house.png?v=10
	async function updateWebView(similar_notes) {
	    const html_links = []
	    for (const n of similar_notes) {
		const ahref = `<a href="#" onclick="webviewApi.postMessage({type:'openNote',noteId:'${n.id}'})">${escapeTitleText(n.title)}</a>`
		html_links.push(ahref);
	    }

	    // TODO wrap in a scrollview
	    
	    await joplin.views.panels.setHtml(panel, `
					<h3>Semantically Similar Notes</h3>
					<div class="container">
						${html_links.join('<br /><br />')}
					</div>
				`);
	
	}

	// TODO not sure what i'm doing with this async/await stuff
	// think I ought to rethink thru the design around this
	const notes = await getAllNoteEmbeddings();
	
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

		Use.load().then(model => {
		    model.embed(document).then(tensor => { // tensor is 512dim embedding of document
			// update our embedding of this note
			const n = notes.get(note.id);
			n['embedding'] = tensor; // update embedding (todo optimize)
			notes.set(note.id, n);
			
			const [sorted_note_ids, similar_note_scores] = search_similar_embeddings(tensor, notes);
			//console.log(sorted_note_ids, similar_note_scores);

			// todo optimize this...
			var sorted_notes = [];
			for (var i = 0; i < notes.size; i++) {
			    //for (const nidx of sorted_note_ids) {
			    const nidx = sorted_note_ids[i];

			    // don't link to ourself (prob always index 0? hopefully...)
			    if (nidx == note.id) {
				continue;
			    }
			    
			    const n: Note = notes.get(nidx);
			    sorted_notes.push(n);
			    console.info(n.title, ": ", similar_note_scores[i]);
			}
			
			updateWebView(sorted_notes);

			// webgl BE requires manual mem mgmt.
			// todo use tf.tidy to reduce risk of forgetting to call dispose

			// TODO
			//tensor.dispose();
		    });				    
		});
		


		
	    } else {
		await joplin.views.panels.setHtml(panel, 'Select a note to see a list of semantically similar notes');
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

	    // so we just need to store the updated embedding in the filesystem
	    // saveEmbeddings();
	    // TODO feels a bit overkill to write the entire json every time a single note changes...
	    // can we use joplin's sqlite to persist to disk?
	});

	// await joplin.settings.onChange( async () => {
	//     updateSimilarNoteList();
	// });

	updateSimilarNoteList('startup');
    },
});
