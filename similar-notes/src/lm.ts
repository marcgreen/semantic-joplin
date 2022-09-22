import * as joplinData from './data';

//const Tf = require('@tensorflow/tfjs');
import * as Tf from '@tensorflow/tfjs';

import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';

import wasmSimdPath from '../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm';
import wasmSimdThreadedPath from '../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm';
import wasmPath from '../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPath,
  'tfjs-backend-wasm-simd.wasm': wasmSimdPath,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmSimdThreadedPath
});
//import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';

//import wasmSimdPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm';
//import wasmSimdThreadedPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm';
// import wasmPath from '/home/marc/semantic_joplin/similar-notes/node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm';

// setWasmPaths({
//   'tfjs-backend-wasm.wasm': wasmPath,
// //  'tfjs-backend-wasm-simd.wasm': wasmSimdPath,
// //  'tfjs-backend-wasm-threaded-simd.wasm': wasmSimdThreadedPath
// });

//require('@tensorflow/tfjs-backend-wasm'); 
import '@tensorflow/tfjs-backend-wasm'; // Add the WASM backend to the global backend registry.
const Use = require('@tensorflow-models/universal-sentence-encoder');


const Log = require('electron-log')

export function enableProd() {
  Tf.enableProdMode(); // not sure the extent to which this helps
  //Tf.ENV.set('WEBGL_NUM_MB_BEFORE_PAGING', 4000);
  //console.log(Tf.memory())
}

export async function loadModel() {
  return await Use.load();
}

export async function setBackend(be) {
  Tf.setBackend(be)
  await Tf.ready(); // any perf issue of keeping this in prod code?
  return Tf.getBackend();
}

// async function loadModel() {
//     // if we already have it saved from disk, load from there
// python ref, but mb helpful: https://stackoverflow.com/questions/69949405/save-and-load-universal-sentence-encoder-model-on-different-machines
//     // otherwise, download from tfhub and save it to disk
// }

// given model from loadModel(),
// and batch as a list of sentences/documents to embed
// run the model on the batch, unstack and sync the tensors
// and return a list of embeddings:
// 1 embedding per element in batch array
// each embedding is itself a 512 element array of floats
export async function embed_batch(model, batch: Array<string>): Promise<Array<number>> {
  let embeddings = [];

  // for testing
  // if (Math.random() > parseFloat(".5")) {
  //   throw 'errmergerd';
  // }

  //const model = await Use.load();
  //Tf.engine().startScope();
  Log.log('embedding batch:', batch.map(n => n.substr(0,100)));
  let tensors: Tf.Tensor = null;
  try {
    tensors = await model.embed(batch);
  } catch (err) {
    Log.error('err embedding batch: ', err);
    //Log.error('err embedding batch: ', err);
    //Log.log('moving to the next batch');
    //return embeddings;
    throw err;
  }
  //console.log(tensors)

  // prob don't want to do this for optimization reasons?
  // (prob faster to compute simlarity all in one go, vs iteratively for each tensor)
  // or maybe we want to untensorize them asap and dispose the tensors?
  const tensors_array = Tf.unstack(tensors);
  //console.log(tensors_array);
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
  
  return embeddings;
  // todo try tf.profile to understand model issue
}


// consider looking at how doc2vec impls this for optimization inspo
export function search_similar_embeddings(embedding: Array<number>, notes: Map<string, joplinData.NoteHeader>) {
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
