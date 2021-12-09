// copied from note graph ui plugin

import joplin from 'api';

export interface Notebook {
  id: string;
  title: string;
  parent_id: string;
}

export async function getNotebooks(): Promise<Array<Notebook>> {
  var allNotebooks = []
  var page_num = 1;
  do {
    var notebooks = await joplin.data.get(['folders'], {
      fields: ['id', 'title', 'parent_id'],
      page: page_num,
    });
    allNotebooks.push(...notebooks.items);
    page_num++;
  } while (notebooks.has_more)

  return allNotebooks;
}

// title helps debug, but theoretically could be out-of-core
export interface NoteHeader {
  id: string;
  parent_id: string;
  title: string;
  linkedToCurrentNote?: boolean;
  embedding?: Array<number>;
  relative_score?: string;
}

export interface Note {
  header: NoteHeader;
  body: string;
}

// Fetch notes
// export async function getNotes(
//   selectedNote: string, maxNotes: number, maxDegree: number): Promise<Map<string, Note>> {
//   if(maxDegree > 0) {
//     return getLinkedNotes(selectedNote, maxDegree);
//   } else {
//     return getAllNotes(maxNotes);
//   }
// }

// /**
//  * Returns a filtered map of notes by notebook name.
//  */
// export async function filterNotesByNotebookName(
//   notes: Map<string, Note>,
//   notebooks: Array<Notebook>,
//   filteredNotebookNames: Array<string>,
//   shouldFilterChildren: boolean,
//   isIncludeFilter: boolean): Promise<Map<string, Note>> {
//   // No filtering needed.
//   if (filteredNotebookNames.length < 1) return notes;

//   const notebookIdsByName = new Map<string, string>();
//   notebooks.forEach(n => notebookIdsByName.set(n.title, n.id))
//   const notebooksById = new Map<string, Notebook>();
//   notebooks.forEach(n => notebooksById.set(n.id, n))

//   // Get a list of valid notebook names to filter out.
//   filteredNotebookNames = filteredNotebookNames.filter(name => notebookIdsByName.has(name));

//   function shouldIncludeNote(parent_id: string): boolean {
//     var parentNotebook: Notebook = notebooksById.get(parent_id)
//     // Filter out the direct parent.
//     console.log("parent ", parentNotebook.title);
//     console.log("there", filteredNotebookNames.includes(parentNotebook.title));
//     if (filteredNotebookNames.includes(parentNotebook.title)) {
//       return isIncludeFilter;
//     }

//     // Filter a note if any of its ancestor notebooks are filtered.
//     if (shouldFilterChildren) {
//       while (parentNotebook !== undefined) {
//         console.log("here")
//         if (filteredNotebookNames.includes(parentNotebook.title)) {
//           return isIncludeFilter;
//         }
//         parentNotebook = notebooksById.get(parentNotebook.parent_id);
//       }
//     }
//     return !isIncludeFilter;
//   }

//   var filteredNotes = new Map<string, Note>();
//   notes.forEach(function(n, id) {
//     if (shouldIncludeNote(n.parent_id)) {
//       filteredNotes.set(id, n);
//     }
//   });

//   return filteredNotes;
//   }

// (re)introduce batch size option
// turn this into pageNotes below; takes a fn
// export async function getAllNotes(): Promise<Map<string, Note>> {
//     var allNotes = []
//     var page_num = 1;
//     do {
// 	// `parent_id` is the ID of the notebook containing the note.
// 	var notes = await joplin.data.get(['notes'], {
// 	    fields: ['id', 'parent_id', 'title', 'body'],
// 	    // for semantic similarity, updated_time seems like an irrelevant ordering.
// 	    // maybe, extract top keywords from current note, FTS those, order by...relevancy?
// 	    // - just a potential optimization to display semantically similar notes faster.
// 	    //   not actually sure how long takes to getAllNotes in practice.
// 	    order_by: 'updated_time',
// 	    order_dir: 'DESC',
// 	    limit: 100,
// 	    page: page_num,
// 	});
// 	allNotes.push(...notes.items);
// 	page_num++;
//     } while (notes.has_more)

//     const noteMap = new Map();
//     for (const note of allNotes) {
// 	noteMap.set(note.id, {id: note.id, title: note.title, parent_id: note.parent_id, body: note.body})
//     }
//     return noteMap;
// }

// we don't page through all notes.
// instead, we buffer a batch via individual fetches.
// well, is GETting one note at a time worse than batching /this/ call
//   and then filtering out already-seen-IDs?
export async function* pageThroughNotesByIDs(forIDs: Array<string>, batch_size: number): AsyncGenerator<Map<string, Note>, void, void> {
  let batch: Map<string, Note> = new Map();
//  for (let i = 0; i < batch_size; i++) {
  for (let id_index = 0; id_index < forIDs.length; id_index++) {
    let note = await joplin.data.get(['notes', forIDs[id_index]], {
      fields: ['id', 'parent_id', 'title', 'body'],
      order_by: 'updated_time',
      order_dir: 'DESC',
      limit: batch_size,
      //page: 1,
    });
      
    let noteHeader: NoteHeader = {id: note.id, title: note.title, parent_id: note.parent_id};//, embedding: null, relative_score: null};
    let noteObj: Note = {header: noteHeader, body: note.body};
    batch.set(note.id, noteObj);
      
    if (batch.size >= batch_size) {
      yield batch;
      batch = new Map();
    }
  }

  yield batch; // the final (partial) batch
}

// Fetches header (without embedding) of every note
export async function getAllNoteHeaders(): Promise<Map<string, NoteHeader>> {
  var allNotes = []
  var page_num = 1;
  do {
    // `parent_id` is the ID of the notebook containing the note.
    var notes = await joplin.data.get(['notes'], {
      fields: ['id', 'parent_id', 'title'],
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
    var links = getAllLinksForNote(note.body);
    noteMap.set(note.id, {id: note.id, title: note.title,
			  parent_id: note.parent_id, links: links})
  }
  return noteMap;
}

// encodeAllNotes // write them to db, return what?

// Fetch all notes linked to a given source note, up to a maximum degree of
// separation.
async function getLinkedNotes(source_id:string, maxDegree:number) : Promise<Map<string, Note>> {

  var pending = [];
  var visited = [];
  const noteMap = new Map();
  var degree = 0;

  pending.push(source_id);
  do {
    // Traverse a new batch of pending note ids, storing the note data in
    // the resulting map, and stashing the newly found linked notes for the
    // next iteration.
    const notes = await getNoteArray(pending);
    visited.push(...pending);
    pending = [];

    notes.forEach(note => {
      // store note data to be returned at the end of the traversal
      const links = getAllLinksForNote(note.body);
      noteMap.set(note.id, {
        id: note.id,
        title: note.title,
        parent_id: note.parent_id,
        links: links});

      // stash any new links for the next iteration
      links.forEach(link => {

        // prevent cycles by filtering notes we've already seen.
        // TODO this check can get expensive for massive graphs
        if(!visited.includes(link)) {
          pending.push(link);
        }
      });
    });

    degree++;

    // stop whenever we've reached the maximum degree of separation, or
    // we've exhausted the adjacent nodes.
  } while(pending.length > 0 && degree <= maxDegree);

  return noteMap;
}

async function getNoteArray(ids:string[]) {

  var promises = ids.map( id =>
    joplin.data.get(['notes', id], {
      fields: ['id', 'parent_id', 'title']
    })
  );

  // joplin queries could fail -- make sure we catch errors.
  const results = await Promise.all(promises.map( p => p.catch(e => e)));

  // remove from results any promises that errored out, returning the valid
  // subset of queries.
  const valid = results.filter(r => !(r instanceof Error));
  return valid;
}

function getAllLinksForNote(noteBody:string) {
  const links = [];
  // TODO: needs to handle resource links vs note links. see 4. Tips note for
  // webclipper screenshot.
  // https://stackoverflow.com/questions/37462126/regex-match-markdown-link
  const linkRegexp = (/\[\]|\[.*?\]\(:\/(.*?)\)/g);
  var match = linkRegexp.exec(noteBody);
  while (match != null) {
    if (match[1] !== undefined) {
      links.push(match[1]);
    }
    match = linkRegexp.exec(noteBody);
  }
  return links;
}
