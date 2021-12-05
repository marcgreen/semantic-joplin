import joplin from 'api';
const Sqlite3 = joplin.plugins.require('sqlite3').verbose();

const Log = require('electron-log')

export function openDB(embeddingsDBPath) {
    let db = new Sqlite3.Database(embeddingsDBPath, (err) => {
	if (err) {
	    console.error(err.message);
	    // TODO what to do for main plugin logic? throw exception? return null?
	    //return null;
	    throw err;
	} else {
	    Log.log('Connected to embeddings db at ', embeddingsDBPath);
	}
    });
    
    return db;
}

export function deleteEmbedding(db, noteID) {
    const stmt = db.prepare("DELETE FROM note_embeddings WHERE note_id = ?");
    stmt.run(noteID).finalize();
    //console.log('deleted ' + noteID);
}

export async function loadEmbeddings(db) {
    Log.log('loading embeddings');
    //    let prom = null;
    let notes = new Map();
    let stmt = null;
    db.serialize(function() {
	db.run("CREATE TABLE IF NOT EXISTS note_embeddings (note_id TEXT PRIMARY KEY, embedding TEXT);");
	//, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);");

	//console.log('table exists');
	
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

export function saveEmbeddings(db, idSlice, embeddings) {
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
