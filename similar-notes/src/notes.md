## resources

tsjs setup: https://www.tensorflow.org/js/tutorials/setup

ml5.js is a higher level ML api built with tf. but they don't seem to have USE, and disabled word2vec for safety reasons

tfjs in electron app: https://github.com/tensorflow/tfjs-examples/tree/master/electron
- notably, can run inference in either backend process or chromium-based frontend env of renderer

Pratik Bhavsar compared the performance of various language models such as BERT, ELMo, USE, Siamese and InferSent

python semantic similarity with tfhub USE: https://colab.research.google.com/github/tensorflow/hub/blob/50bbebaa248cff13e82ddf0268ed1b149ef478f2/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb
- compare this to top2vec's impl
- is tf.Session() relevant at all?

tsjf + USE for sentence similarity: https://github.com/jinglescode/demos/blob/master/src/app/components/nlp-sentence-encoder/nlp-sentence-encoder.component.ts
- can steal their helper funcs (dot, similarity, cosine_sim_matrix)

joplin backlinks plugin: https://github.com/ambrt/joplin-plugin-referencing-notes/blob/master/src/index.ts
- registers callback functions in onStart

gentle intro to doc2vec and word2vec: https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e
> The thing with this kind of unsupervised models, is that they are not trained to do the task they are intended for. E.g, word2vec is trained to complete surrounding words in corpus, but is used to estimate similarity or relations between words.

## joplin plugin notes

from api/joplin.d.ts, it says we can't bundle native pkgs with a plugin bc of cross-platform. does this mean i can't use tensorflow with native cpp bindings? and need pure js version?

from api/joplinplugins.d.ts, i think i can store the LM in dataDir() loc

use joplinviewspanels.d.ts to create View to display list of semantically similar notes

workspace service to ge currently selected note, and when note content changes

package.json implies joplin uses webpack for building dist (informs tensorlfow.js approach)

plugin build error: can't resolve fs
- uninstalled tfjs-node
- solved by adding following to package.json. not sure if this will be a problem for publishing plugin?
```
  "browser": {
      "fs": false
  }
```

### arch thoughts

- have button in plugin settings to train the model on all nodes
- on note view, query the model for top 10 similar notes
  - check if model already trained, and show msg to user if not to train it
- when note changed, remove old note from model, and add new note
  - how slow might this be?

### caveats/limitations

- need to recreate the model if # of notes significant changes
  - would it help to recreate model over time, too, after many things are removed/added?
- not sure how many notes would be needed to benefit from a different LM (eg BERT)
- not sure how well this works for smaller # of notes. development was done with a corpus of 800 notes (6mb text)
- no support for attachments yet

### future work

- setting to adjust how many similar notes are displayed
- setting to exclude specified notebooks from being included 
- add option to remove linked notes from results (since they are obv already known/accounted for by user)
- viz note similarities in 2d or 3d
- optionally include note's tags in the embeddings (test how this changes results per note in practice)
- - see gensim impl here: https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e

## tfjs notes

tfjs has 'universal sentence encoder' LITE (converted to tsjs GraphModel), which is based on Transformer arch with 8k word piece vocab. wonder if this implies worse performance compared to top2vec's USE impl.


## top2vec notes

### how top2vec uses USE for document similarity

- _embed_documents uses numpy vstack, array, and sklearn’s normalize
  - uses tfhub’s USE embedding model via tfhub.load() before this ^ to embed in batches of 500
  - where is it separating per doc? i assume train_corpus is array of docs, and loaded model takes care of it?
- i’m using top2vec’s embedding model with universal-sentence-encoder - would i see worse performance if i use USE’s embedding model?
  - top2vec’s default tokenizer uses gensim’s simple_preprocesses and parsing.perprocessing strip_tags
- top2vec uses sklearn’s feature_extraction.text CountVectorizer to preprocess vocab
  - maybe not needed if we don’t have word embeddings, and only document?
  - if not using embedding model of USE, think i do need this

### seach_documents_by_document

this is what we are trying to do with our plugin.

https://github.com/ddangelov/Top2Vec/blob/76513f488df27caf7c1befd463dcf4ebab844018/top2vec/Top2Vec.py#L2271

what exactly are the indexes top2vec uses? says it speeds up the searching.

converts doc to a vec, then searches vectors:
```
    def _search_vectors_by_vector(vectors, vector, num_res):
        ranks = np.inner(vectors, vector)
        indexes = np.flip(np.argsort(ranks)[-num_res:])
        scores = np.array([ranks[res] for res in indexes])

        return indexes, scores
```

inner refers to inner product of two arrays
flip reverses the order of elements in an array along the given axis. shape preserved, elements reordered.
argsort returns the indices that would sort an array

is it this easy to do in js? is there numpy equiv?
- maybe tensorflow.js itself?
- 