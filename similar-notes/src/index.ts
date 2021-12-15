import joplin from 'api';
import * as Ui from './ui';
import * as Lm from './lm';
import * as Db from './db';
import * as joplinData from './data';
import * as joplinSettings from './settings';

const Log = require('electron-log')
//Object.assign(console, Log.functions);

const Fs = joplin.plugins.require('fs-extra');
const Path = require('path');

Lm.enableProd();

// partial todo list
// - optimize if necessary (don't unstack tensors, *Sync() to *(), fix all await/async/promises)
// - - save USE model to disk so it's not redownloaded every time
// - - recompute embedding (and ALL similirities if we can limit cpu/gpu and do in bg) via event queue
// - ought to use event api for tracking note creation/updates/deletion
// - clean things up
// - - probably some large refactors doable, now that I understand flow better
// - manually test some edge cases?
// - compare semantic similarity results with full USE model, vs this USE lite model

// next patch:
// - update readme to exlpain settings (if no progress at startup after while, set backend to cpu and try again)
// - use tf.data.generator instead of my own generator?
//     also has a batch function
//   - use async versions to impl throttling?
//     - can also abort onNotechange with this?


// future patch:
// version_bump script that also prepends to changelog via git commits
// - maybe next time
// race conditions when deleting notes?
// - aborting computations upon note change would fix this
// - use tidy in case there's a memleak?


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

function notes2docs(notes: Array<joplinData.Note>) {
  console.log('notes: ', notes);
  let docs = [];
  for (const n of notes) {
    //docs.push(n.title);
    docs.push(n.header.title + "\n" + n.body);
  }
  return docs;
}

// do the exact same as syncEmbeddings, but don't change the webview output at all
// (specifically used to hide sync at the start of every note change)
async function silentSyncEmbeddings(model, db, panel): Promise<Map<string, joplinData.NoteHeader>> {
  // pass in a mock panel? turn off smt in Ui? if end up passing it a param, then don't need this func anymore
  // todo
  return null;
}

// creates new embeddings batch_size at a time
// deletes embeddings from db for notes no longer in joplin
// returns map of note id to note header (with embedding)
async function syncEmbeddings(model, db, panel): Promise<Map<string, joplinData.NoteHeader>> {
  let progressHTML = '<center><i>Computing/loading embeddings</i></center>';
  await Ui.updateHTML(panel, progressHTML);
  
  // id->noteHeader w/ embeddings based on loaded + newly created
  // we fill this in as we go and return this at the end
  const canonNoteHeaders = await joplinData.getAllNoteHeaders();
  const canonNoteIDs = [...canonNoteHeaders.keys()];

  progressHTML += `<br /><br />Total # notes: ${canonNoteIDs.length}`;
  await Ui.updateHTML(panel, progressHTML);
  
  // try loading saved embeddings first
  // determine which notes don't yet have embeddings, compute and save those

  const savedEmbeddings = await Db.loadEmbeddings(db); // map of noteID to 512dim array
  const knownIDs = [...savedEmbeddings.keys()];
  console.log('savedEmbeddings:', savedEmbeddings);

  progressHTML += `<br />Saved # embeddings: ${knownIDs.length}`;

  
  // todo use event queue to handle this better
  // delete notes from DB that are no longer in joplin proper
  const deletedIDs = knownIDs.filter(id => !canonNoteIDs.includes(id));
  console.log('note embeddings to delete from db: ', deletedIDs);
  for (const nid of deletedIDs) {
    Db.deleteEmbedding(db, nid);
    savedEmbeddings.delete(nid);
  }

  // savedEmbeddings has id->{embedding} of relevant loaded embeddings
  for (const [nid, dict] of savedEmbeddings) {
    const n = canonNoteHeaders.get(nid);
    n.embedding = dict.embedding;
    canonNoteHeaders.set(nid, n);
  }

  await Ui.updateHTML(panel, progressHTML);

  // batch encode the unembedded notes
  // let remainingNoteHeaders = new Map();
  // split the remaining notes needing to be embedded from allNotes,
  //   based on what was loaded
  const unembeddedIDs = canonNoteIDs.filter(id => !knownIDs.includes(id));
  
  progressHTML += `<br />Remaining # embeddings: ${unembeddedIDs.length}`;

  
  // now that we page, maybe rewrite to not bother iterating over ids
  // and instead just filter the loaded/remaining from canon? todo
  // for (const nid of unembeddedIDs) {
  //   remainingNoteHeaders.set(nid, canonNoteHeaders.get(nid));
  // }

  const batch_size = Math.max(1, await joplin.settings.value('SETTING_BATCH_SIZE'));
  const num_batches = Math.floor(unembeddedIDs.length/batch_size);
  const remaining = unembeddedIDs.length % batch_size;
  Log.log('batches to run ', num_batches, ' ', remaining);

  progressHTML += `<br /><br />Batch Size: ${batch_size} notes`;
  progressHTML += `<br /># full batches: ${num_batches}`;
  progressHTML += `<br /># notes in final partial batch: ${remaining}`;
  await Ui.updateHTML(panel, progressHTML);

  // process the remaining notes
  // use a generator around joplin data api to batch for out-of-core
    // optimization: change algo based on how many notes there are.
    // ie, loop through all when there are a lot to embed
    // and 1 at a time when there are only a few.
    // currently doing latter all the time.
    // 
  //let createdEmbeddings = []
  let i = 0;
  for await (const noteMap of joplinData.pageThroughNotesByIDs(unembeddedIDs, batch_size)) {
    console.log('starting batch ', i);
    const documentBatch = notes2docs([...noteMap.values()]);
    const idBatch = [...noteMap.keys()];
    
    Log.log('creating embeddings');
    //const tensors = await model.embed(['test']);
    //let embeddings = [];
    
    progressHTML += "<br />";
//  for (let i = 0; i < num_batches; i++) {
    //const slice = remaining_documents.slice(i*batch_size, (i+1)*batch_size);
    //const idSlice = unembeddedIDs.slice(i*batch_size, (i+1)*batch_size);
    
    //console.log(i, slice);
    let e = null;
    let startTime = new Date().getTime();
    try {
      e = await Lm.embed_batch(model, documentBatch);
    } catch (err) {
      Log.log('moving to the next batch');
      continue;
    }
    // originally designed this way to accommodate model crashing on large input, 
    // but didn't end up figuring out how to force commit to DB before moving on,
    // so ought to be refactored...
    Db.saveEmbeddings(db, idBatch, e);

    let endTime = new Date().getTime();
    let execTime = (endTime - startTime)/1000;
    //console.log('e should be array, first ele first embedding: ', e)
    //embeddings = embeddings.concat(e);
    //console.log('done ', i);

    // track our newly created embeddings in the map we return
    // ...this is the third/4th time we iterate over this batch...
    // ...todo...
    for (const nid of noteMap.keys()) {
      const h = canonNoteHeaders.get(nid);
      h.embedding = e.shift(); // iterating notes in order
      canonNoteHeaders.set(nid, h);
    }

    Log.log('finished batch ' + i, execTime + ' seconds elapsed');
    //console.log(Tf.memory(), Tf.engine(), Tf.env());

    progressHTML += `<br />Finished batch ${i+1} in ${execTime} seconds`;
    await Ui.updateHTML(panel, progressHTML);
    
    i++;
  }
  // if (remaining > 0) {
  //   const slice = remaining_documents.slice(num_batches*batch_size);
  //   const idSlice = unembeddedIDs.slice(num_batches*batch_size);
  //   //console.log(slice);
  //   const e = await Lm.embed_batch(model, slice);
  //   Db.saveEmbeddings(db, idSlice, e);
  //   embeddings = embeddings.concat(e);
    
  //   progressHTML += `<br />Finished final batch`;
  //   await Ui.updateHTML(panel, progressHTML);
  // }
  //const tensors = await model.embed(remaining_documents);
  //console.log('created', num_batches, ' ', remaining);

  
  //console.log(embeddings);
  // const keys = [....keys()];
  //    for (const nid of unembeddedIDs) {
  
  // embeddings is array of created embeddings
  // unembeddedIDs is array of noteIDs, same order+length as embeddings
  // for (let i = 0; i < unembeddedIDs.length; i += 1) {
  //   const nid = unembeddedIDs[i];
  //   let n = remainingNoteHeaders.get(nid);
  //   n.embedding = embeddings[i];
  //   //console.log(n);
  //   canonNoteHeaders.set(nid, n);
  //   //embedding_map[allNotes[i].id] = tensors_array[i];
  // }
  //console.log('all notes with embeddings:', allNotes);

  return canonNoteHeaders;
}


async function propagateTFBackend(event) {
  const tfjsBackend = await joplinSettings.getSelectedBackend();
  const be = await Lm.setBackend(tfjsBackend);
  Log.log('tensorflow backend: ', be);
}

// orig notes (go through these)
// need to create initial embeddings while fetching ntoes in batches of 100 (default)
// this sounds perfect for async / promises
// currently, code does:
// onstart
// openDB
// model = await Use.load()
// notes = await getAllNoteEmbeddings()
//
//   allNotes = await getAllNotes() // only want to get 100 here,
//   loadEmbeddings() // fine to load all in mem since even with
//     100,000 notes * 4byte(?)*512 floats is less than 256mb
//   given all loaded embeddings and all notes, filter the ones we don't know yet
//   - this needs to be reworked to accommodate only loading 100 at a time.
//       maybe just keep track of them per batch of 100, and then after loading all 100,
//       we know the id of the ones to create embeddings for...but is there an easy way
//       to load those selectively again?
//   - instead of just knowing the id, just keep in memory the ones we don't have
//       embeddings for, and toss the rest. and create embeddings for these ones.
//       ie, move the filter to the batch for loop
//   deletion of ids we have embeddings for but not in allNotes -> can't do in batches
//     bc they aren't sorted (we don't know if they will show up later)
//   - could accumulate these ids during batch and delete them all at the end in a loop
//   notes2doc in batch
//   calc # of batches
//   - can we know the # of total notes before batching? do we need to iterate through
//     all notes once to find that, and iterate again to batch? presumably the first
//     iteration would go orders faster since the encoding is the slow bit. can check
//     the joplin api to see if this is readily available in a call
//       could then technically accumulate deletion ids in this first iteration
//         well we could just select only the id column from the db from this first
//         iteration. but why not just select all note IDs in getAllNotes first, too?
// think this through ->
//   batch the full size
//     - move filter for unknown and accumulate over-known ids here.
//       how slow is it to create and dispose tensors? is that relevant to this?
//       ie, would we want to hold onto them in memory for cross product recalcs?
//         especially when sub'd to event model and update scores for all notes against
//         all other notes (NxNx512). maybe we need to do this in batches, too. but also
//         on-demand if the user switches to a note if it takes more than a sec or two?
//         this makes less worthwhile to do NxNx512 bc any add/edit/delete will trigger
//         the computation delay that I'm trying to get rid of. worthwhile nonetheless?
//     calc the batch, unstack the tensors, verb the tensors,
//       accumulate values, save values to db
//   batch the remaining
//   from loaded and created embeddings, create full note object
//   - want to only track id and embedding (and maybe title), not the body

// i wonder ratio of successful:error users of the plugin so far? afaik, incl me,
// it's 1:2.

// one result on google says roman's error is caused from too much memoryin use,
//   so laurent's idea might fix that. maybe allNotes takes up too much space for
//   tfjs to compute the embedding of one of his larger notes?
//   wonder what his largest notes sizes are.
//   what if this doesn't fix it though? then would need to break up large notes
//     into a series of smaller notes and average their embeddings or smt?
//     or at least detect it and skip the note for now (distinguish in note list UI too)
// - https://github.com/tensorflow/tfjs/issues/1644#issuecomment-498322722
// added more logging around the embedding to hopefully catch the error he posted

// about whitewall's...what happens after calcing the # batches?
// - async func def for embd
// - starts batch loop
//   - 2 array slices, start a timer, call embed(), then update html
// could add console.info statements to verify where it's hanging, but it's probably
// in the call the embed the first 100 docs
// - could encode docs one at a time to see if either that fixes it or at least
//   reveals a particular note it consistent hangs on
// could print out doc body length of each batch before embedding
// hopefully solved by cpu backend




joplin.plugins.register({
  onStart: async function() {
    await joplinSettings.registerSettings();
    joplin.settings.onChange(event => propagateTFBackend(event))
    propagateTFBackend(null); // with default value
    
    const selectNotePromptHTML = '<br /><i><center>Select a note to see similar notes</center></i>'

    // Create the panel object
    const panel = await joplin.views.panels.create('semanticlly_similar_notes_panel');
    await joplin.views.panels.onMessage(panel, async (message) => {
      await joplin.commands.execute("openNote", message.noteId)
    });

    const pluginDir = await joplin.plugins.dataDir();
    const embeddingsDBPath = Path.join(pluginDir, 'embeddings.sqlite');
    Log.log('Checking if "' + pluginDir + '" exists:', await Fs.pathExists(pluginDir));

    const db = Db.openDB(embeddingsDBPath);

    await Ui.updateHTML(panel, '<center><i>Downloading model from Tensorflow Hub</i></center>')
    const model = await Lm.loadModel();
    //console.log(Tf.memory())
    console.log(model);
    
    // this syncs the embeddings in the db given the info from joplin,
    // - creates new embeddings from notes not yet in embedding db
    // - deletes embeddings of notes no longer in joplin
    // returns map of id to note headers (with embedding)
    let noteHeaders = await syncEmbeddings(model, db, panel);
    // todo move part of this function inside updateSimilarNoteList
    //  so that new note title names are accurate. but don't want to relaod
    //  everything from DB
    // todo: don't include body in this list

    await Ui.updateHTML(panel, selectNotePromptHTML);

    // if reEmbed,
    //   this will compute the embedding for the selected note,
    //   update the var in which we store all notes,
    //   and save the new embedding to the db.
    // regardless of reEmbed, this will:
    //   compute the similarities to all other notes,
    //   and display them in sorted order in the WebView
    // todo could conditionally recompute similarities, too
    async function updateSimilarNoteList(updateType: string, reEmbed: boolean) {
      console.log('updating bc: ', updateType)
      // Get the current note from the workspace.
      const note = await joplin.workspace.selectedNote();

      // refresh our cache of noteheaders (with updated titles, eg)
      // will this take too long?
      noteHeaders = await syncEmbeddings(model, db, panel);

      // Keep in mind that it can be `null` if nothing is currently selected!
      if (note) {
	console.log('selected note title:\n', note.title);

	await Ui.updateHTML(panel, 'Computing similarities...');

	let embedding = null;
	let noteHeader = noteHeaders.get(note.id);
	
	// if there is no note object, it's a new note, so create note obj
	// and "re"Embed it
	if (!noteHeader) {
	  reEmbed = true;
	  noteHeader = {id: note.id, title: note.title,
		     parent_id: note.parent_id, //body: note.body,
		     embedding: null, // will be set in a sec
		     relative_score: null // will be set in a sec
		    }
	}
	
	if (reEmbed) {
	  const n: joplinData.Note = {header: noteHeader,
				      body: note.body};
	  const [document] = notes2docs([n]);
	  try {
	    [embedding] = await Lm.embed_batch(model, [document])
	  } catch (err) {
	    Log.log("err embedding note, that's weird");
	    await Ui.updateHTML(panel, "Error embedding note: " + note.title + ' - ' + note.id);
	    return;
	  }

	  // update our embedding of this note
	  noteHeader.embedding = embedding;
	  noteHeaders.set(note.id, noteHeader);

	  // persist the calculated embedding to disk
	  // todo anyway to detect if the change doesn't make it?
	  //  eg if pc lost power between the joplin note saving to disk
	  //  and this func saving the corresponding new embedding,
	  //  then results would be off until next time user edits this note
	  // - could compare timestamp of last note change with timestamp
	  //   of last embedding change on startup
	  //console.log('test before save');
	  Db.saveEmbeddings(db, [note.id], [embedding]);
	} else {
	  embedding = noteHeader['embedding'];
	}
	  
	  //console.log('noteHeaders w/o embeddings ', [...noteHeaders.values()].filter(h => !h.embedding));
	//console.log('embeddings: ', embedding);
	const [sorted_note_ids, similar_note_scores] = Lm.search_similar_embeddings(embedding, noteHeaders);
	//console.log(sorted_note_ids, similar_note_scores);

	// todo optimize this...
	// - keep things as tensors? but when to dispose then? no onShutdown
	// - do large tensor multiplication of all note sims at once?
	//   could do for 1:N note sims, but maybe also N:N?
	let sorted_notes: Array<joplinData.NoteHeader> = [];
	for (let i = 0; i < noteHeaders.size; i++) {
	  //for (const nidx of sorted_note_ids) {
	  const nidx = sorted_note_ids[i];

	  // don't link to ourself (prob always index 0? hopefully...)
	  if (nidx == note.id) {
	    continue;
	  }
	  
	  const n = noteHeaders.get(nidx);
//	  console.log(n);
	  n['relative_score'] = (similar_note_scores[i]*100).toLocaleString(undefined, {maximumSignificantDigits: 2});
	  sorted_notes.push(n);
	  //console.info(n.title, ": ", similar_note_scores[i]);
	}
	
	Ui.updateUIWithNoteList(panel, sorted_notes);

	// webgl BE requires manual mem mgmt.
	// use tf.tidy to reduce risk of forgetting to call dispose

	//model.dispose();
      } else {
	await Ui.updateHTML(panel, selectNotePromptHTML);
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

    // bugs if this is false? todo
    updateSimilarNoteList('startup', true);
  },
});

