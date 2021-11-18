import joplin from 'api';

const Tf = require('@tensorflow/tfjs');
const Use = require('@tensorflow-models/universal-sentence-encoder');

//tf.enableDebugMode()

joplin.plugins.register({
    onStart: async function() {
	console.info('Hello world. Test plugin started!');
	console.log(Tf.getBackend());

	// Load the model.
	async funcion loadModel() {

	}
	

	console.info('HelloHello');

	// embed all notes into the model (on button click in settings, if possible)
	// - what kind of preprocessing is needed?
	// save the model to disk
	// load the model (or show a message to user to go to settings)
	// query the model
	// - need to compare against every note afaik. wonder how slow this will be.

	async updateWebView(similar_notes) {
	    return;
	}

	// TODO look at how doc2vec impls this
	function search_similar_embeddings(tensor) {
	    // vectors = ??? need access to the model's embeddings for all documents (as tensors?)
            //ranks = np.inner(vectors, vector)
            //indexes = np.flip(np.argsort(ranks)[-num_res:])
            //scores = np.array([ranks[res] for res in indexes])

            //return indexes, scores

	}

	// queries model for similar notes
	// updates webview with results
	async function updateSimilarNoteList() {
	    // Get the current note from the workspace.
	    const note = await joplin.workspace.selectedNote();

	    // Keep in mind that it can be `null` if nothing is currently selected!
	    if (note) {
		//console.info('note is:', note);

		Use.load().then(model => {
		    // Embed an array of sentences.
		    const sentences = [
			'Hello.',
			'How are you?'
		    ];
		    document = note.title + "\n" + note.body;
		    model.embed(embed).then(embeddings => {
			// `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
			// So in this example `embeddings` has the shape [2, 512].
			embeddings.print(true /* verbose */);
			console.info(embeddings);

			similar_note_indexes = search_similar_embeddings(embeddings)

			// todo only send list of...how to refer to joplin note s.t. it can be displayed?
			similar_notes = similar_note_indexes // todo
			webviewUpdate(similar_notes)
		    });
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
