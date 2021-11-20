import joplin from 'api';

const Tf = require('@tensorflow/tfjs');
const Use = require('@tensorflow-models/universal-sentence-encoder');

//tf.enableDebugMode()


	// embed all notes into the model (on button click in settings, if possible)
	// - what kind of preprocessing is needed?
	// save the model to disk
	// load the model (or show a message to user to go to settings)
	// query the model
	// - need to compare against every note afaik. wonder how slow this will be.

// code borrowed from joplin link graph plugin
// todo add types back in lol
async function getAllNotes() {
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

    return allNotes;
    // this might be useful when filtering out linked note
    // const noteMap = new Map();
    // for (const note of allNotes) {
    //   var links = getAllLinksForNote(note.body);
    //   noteMap.set(note.id, {title: note.title, parent_id: note.parent_id, links: links})
    // }
    // return noteMap;
}

// TODO look at how doc2vec impls this
function search_similar_embeddings(tensor, tensors) {
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
    //const num_tensors = tensors.arraySync()[0].length
    //Tf.unstack(tensors).forEach(t => t.print(true));
    // todo extend tensor to same dim as tensors, and do mult in 1 op, vs forEach
    //const flipped = Tf.transpose(tensor);
    //Tf.unstack(tensors).forEach(t => scores.push(Tf.dot(tensor, t)));

    const ts = Tf.unstack(tensors);
    //console.log(ts.length)
    for (const t of ts) {
	const score = parseFloat(Tf.dot(tensor, t).dataSync());
	//tensor.print(true);
	//t.print(true);
	//score.print(true);
	//console.log(score.dataSync()); // not a tensor, just an array32Float
	//console.log(parseFloat(score.dataSync())); // normal js float
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
    
    const {values, indices} = Tf.topk(scores, ts.length);
    const syncedIndices = indices.arraySync();
    // console.log(indices);
    // console.log(syncedIndices);
    // for (const i of indices) {
    //  	i.print();
    // }

    values.print();
    indices.print();
    
    // sort the scores, but keep track of the index they were in? aka np.flip + np.argsort I think
    // then retrieve scores
    
    return [syncedIndices, values.arraySync()];
    
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
    //console.log('notes: ', notes);
    const docs = notes.map(note => note.title + "\n" + note.body);
    return docs;
}


// dataDir for saving/loading embeddings: https://joplinapp.org/api/references/plugin_api/classes/joplinplugins.html

joplin.plugins.register({
    onStart: async function() {
	await Tf.ready(); // any perf issue of keeping this in prod code?
	console.info('tensorflow backend: ', Tf.getBackend());

	// todo see toc plugin for basics of adding/styling/interactivizing FE UI element
	//   https://joplinapp.org/api/tutorials/toc_plugin/
	//
	// also the Favorites plugin does smt similar to what I envison wrt UI element
	//   https://emoji.discourse-cdn.com/twitter/house.png?v=10
	async function updateWebView(similar_notes) {
	    return;
	}

	// looking at the QnA model code, I anticipate I'll want to save the embeddings.
	// ok, so the large-scale compare embed
	// but before investing in that, want to make sure I can get the use of
	//   embeddings working with small #
	// this might be analagous to top2vec's model save/load/create model mechanism?
	// queries model for similar notes
	// updates webview with results
	async function updateSimilarNoteList(updateType: string) {
	    console.info('updating bc: ', updateType)
	    // Get the current note from the workspace.
	    const note = await joplin.workspace.selectedNote();
	    console.info('selected note title:\n', note.title);

	    // getting all notes here just to prove this works.
	    // then do this when user clicks button in setting, and store to global var or smt
	    const all_notes = await getAllNotes();

	    // Keep in mind that it can be `null` if nothing is currently selected!
	    if (note) {
		const [document] = notes2docs([note]);
		//console.info('document:\n', document);

		Use.load().then(model => {
		    model.embed(document).then(tensor => {
			//console.info(tensor);
			// `tensor` is a tensor consisting of the 512-dimensional embeddings for each sentence.

			// TODO update idex with embedding for this note, keyed on note.id

			const all_documents = notes2docs(all_notes);
			model.embed(all_documents).then(all_tensors => {
			    //console.info(all_tensors);
			    // all_tensors.print(true);
			    // should be tensor with n elements, each 512 dim tensor
			    const [similar_note_indexes, similar_note_scores] = search_similar_embeddings(tensor,
													  all_tensors);

			    for (const nid of similar_note_indexes) {
				console.info(all_notes[nid].title, ": ", similar_note_scores[nid]);
			    }
			    // todo only send list of...how to refer to joplin note s.t. it can be displayed?
			    // todo it def needs to be clickable. ideally like the main note scrolllist. if not, then like
			    //   backlink list hyperlinks
			    
			    //similar_notes = similar_note_indexes // todo
			    //webviewUpdate(similar_notes)

			    // webgl BE requires manual mem mgmt.
			    // todo use tf.tidy to reduce risk of forgetting to call dispose

			});
			// TODO
			//tensor.dispose();
			//all_tensors.dispose();
		    });				    
		});
		


		
	    } else {
		// TODO message in WebView that no note is selected
		//console.info('No note is selected');
	    }
	}

	// This event will be triggered when the user selects a different note
	await joplin.workspace.onNoteSelectionChange(() => {
	    updateSimilarNoteList('note selection');
	});

	// This event will be triggered when the content of the note changes
	await joplin.workspace.onNoteChange(() => {
	    // TODO re-embed note (overwrite old embedding)
	    // time how long it takes for a lenghty note. what happens if user closes joplin during this?
	    
	    updateSimilarNoteList('note change');
	});

	// await joplin.settings.onChange( async () => {
	//     updateSimilarNoteList();
	// });

	// Also update when the plugin starts
	updateSimilarNoteList('on start');
	
    },
});
