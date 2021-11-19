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




// TODO look at how doc2vec impls this
function search_similar_embeddings(vector, vectors) {
    // top2vec's impl in python for 1 to many string similarity search
    //ranks = np.inner(vectors, vector)
    //indexes = np.flip(np.argsort(ranks)[-num_res:])
    //scores = np.array([ranks[res] for res in indexes])

    //return indexes, scores

    // vectors = ??? need access to the model's embeddings for all documents (as tensors?)

    // todo move to helper function (search_similar_embeddings)
    // todo this is np.inner
    var ranks = [];
    for (let i = 0; i < vectors.length; i++) {
	ranks.push(dotProduct(vector, vectors[j]));
    }

    // sort the ranks, but keep track of the index they were in? aka np.flip + np.argsort I think
    // then retrieve scores
    
    return [indexes, scores]
    
    // for vectors to vectors: (todo replace with np.inner, still)
    // var scores = [];
    // for (let i = 0; i < vector1.length; i++) {
    // 	for (let j = 0; j < vector2.length; j++) {
    // 	    scores.push(dotProduct(vector1[i], vector2[j]));
    // 	}
    // }

}

// Calculate the dot product of two vector arrays.
const dotProduct = (xs, ys) => {
    const sum = xs => xs ? xs.reduce((a, b) => a + b, 0) : undefined;

    return xs.length === ys.length ?
	sum(zipWith((a, b) => a * b, xs, ys))
	: undefined;
}

// zipWith :: (a -> b -> c) -> [a] -> [b] -> [c]
const zipWith =
      (f, xs, ys) => {
	  const ny = ys.length;
	  return (xs.length <= ny ? xs : xs.slice(0, ny))
	      .map((x, i) => f(x, ys[i]));
      }


joplin.plugins.register({
    onStart: async function() {
	console.info('Hello world. Test plugin started!');
	console.log(Tf.getBackend());

	// Load the model.
	async updateWebView(similar_notes) {
	    return;
	}

	// looking at the QnA model code, I anticipate I'll want to save the embeddings.
	// ok, so the large-scale compare embed
	// but before investing in that, want to make sure I can get the use of
	//   embeddings working with small #
	// this might be analagous to top2vec's model save/load/create model mechanism?

	function notes2docs(notes) {
	    const documents = notes.map(note => note.title + "\n" + note.body);
	    return documents;
	}
	
	// queries model for similar notes
	// updates webview with results
	async function updateSimilarNoteList() {
	    // Get the current note from the workspace.
	    const note = await joplin.workspace.selectedNote();

	    // Keep in mind that it can be `null` if nothing is currently selected!
	    if (note) {
		document = notes2docs([note]);
		console.info('document:\n', document);

		Use.load().then(model => {
		    const embedding = model.embed(document).arraySync();
		    // todo can use async version via .then. do we want to here?
		    // todo if this errors, make it an array of a single document
		    
		    console.info(embeddings);
		    // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.

		    // TODO update idex with embedding for this note, keyed on note.id

		    // getting all notes here just to prove this works.
		    // then do this when user clicks button in setting, and store to global var or smt
		    const all_notes = await joplin.workspace. // LEFT OFF HERE
		    const all_documents = notes2docs(all_notes);
		    const all_embeddings = model.embed(all_documents).arraySync(); // .then TODO

		    const [similar_note_indexes, similar_note_scores] = search_similar_embeddings(embedding,
												  all_embeddings);


		    // todo only send list of...how to refer to joplin note s.t. it can be displayed?
		    // todo it def needs to be clickable. ideally like the main note scrolllist. if not, then like
		    //   backlink list hyperlinks
		    
		    //similar_notes = similar_note_indexes // todo
		    //webviewUpdate(similar_notes)
		});
		


		
	    } else {
		//console.info('No note is selected');
	    }
	}

	// This event will be triggered when the user selects a different note
	await joplin.workspace.onNoteSelectionChange(() => {
	    updateSimilarNoteList();
	});

	// This event will be triggered when the content of the note changes
	// as you also want to update the TOC in this case.
	await joplin.workspace.onNoteChange(() => {
	    updateSimilarNoteList();
	});

	// await joplin.settings.onChange( async () => {
	//     updateSimilarNoteList();
	// });

	// Also update when the plugin starts
	updateSimilarNoteList();
	
    },
});
