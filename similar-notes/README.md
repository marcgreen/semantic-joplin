# Semantically Similar Notes

## Summary

The intent of this plugin is to help the joplin user find notes they have that might be relevant to the one they are editing. We plan to filter out linked notes from results, as an explicit linking implies the user is already aware of the relevance. This filtering is not yet implemented, however.

Disclaimer: This plugin was written during a 2-week hackathon, without prior javascript or tensorflow experience, and the code quality reflects this. PRs are welcome, but I do plan to refactor things with the knowledge I now have after getting it to the current state.

## Potential Use Cases

### Suggesting relevant notes from your corpus to link to

TODO insert content from joplin forum post

### Find duplicate and near-duplicate notes

If there's a lot of words in common between the notes, they'll likely be very semantically similar to each other.

### Crude Semantic Search

By creating a new note with your search query, you can retrigger similarity calculations to perform your semantic search.

Anecdotally, I've only /meaningfully/ used this a total of one time so far to ignore word endings and find notes based on words that had the same "stem". Which makes sense, since words with the same stem are likely going to be semantically similar to each other.

Example: "analogy" and "analogies" won't find each other in keyword search, but they can find each other in semantic search.



### On First Use

TODO: document cpu/gpu and batch size options and how they can help in this step

On first startup, this plugin will calculate embeddings for all of your notes. **This can take some minutes** (less than 2min for my 800 notes totalling 5mb not including attachments on my desktop computer). It saves these embeddings to disk and loads from there on subsequent startups. Of course, if you work on notes outside of where this plugin runs, then upon sync, it will need to calculate embeddings for those notes too. It's future work to make this happen upon user request vs automatically at startup.

**If you have notes larger than a few hundred KBs, this might take a long time.** Notes larger than 1MB might even make the model hang, I'm not entirely sure yet. The largest notes in my corpus are a few hundred KB. I think calculating the embeddings is exponential on note size, or at least superlinear.

Please let me know your system specs (cpu+ram, gpu+vram) and how big your largest notes are (excl. attachments) if it takes longer than 10min to do a single batch during the initial embeddings.


## Semanticness

We use tensorflow.js's [Universal Sentence Encoder Lite](https://github.com/tensorflow/tfjs-models/tree/master/universal-sentence-encoder) to encode each note into a 512 dimension vector, called an embedding. It's called a 'sentence encoder', but it seems to work for longer strings of text, too (eg, see doc2vec and top2vec). We use the dot product of two embeddings to determine similarity.

It might be the case that better results could be achieved with a different model (eg mobileBERT, or BERT), or perhaps an entirely different approach, like one that does a simple keyword search instead (assuming a way to determine the right keywords to use, perhaps via topic extraction).

### Quality of Results

Sometimes the results of the semantic similarity computation are confusing or questionable. Consider that the model is comparing what it thinks the meaning of the words you're using in the selected note against all other notes. So there are three variables at play: the words used in the selected note, the words used in the note being compared against, and the model's understanding of our language. A potential nuance of this is that since each embedding is a fixed size, longer inputs presumably mean each token in the input is given less weight. I'm not sure the extent this affects similarity scores.

Testing the model against my own corpus of notes, I am satisified enough with its performance. But I would certainly be interested in experimenting with different models (and different approaches), like the ones mentioned in Future Work below.

## Speed

Because we're using a dense neural network, computation of the embeddings is relatively slow. However, even after the embeddings are computed, just computing the 512x512 dot product to determine similarity takes a noticeable amount of time. This isn't surprising, as a few times during development I chose to forego obvious optimizations in order to finish this first version of the plugin within the time constraints I was given. I have ideas on how to optimize these computations, but for now, it seems to take a couple seconds to show results every time you switch notes. Still, fast enough to be useful, in my opinion.


## Debugging

Logs can be found at `%APPDATA%\Roaming\@joplin\app-desktop\logs` and also I think `~\.config\joplin-desktop\log.txt`

## Caveats/Limitations

- english only, and USE lite has 'only' an 8000 word vocabulary
- requires internet connection to download the USE model every startup. would like to save it locally after first download
- no support for attachments yet

## Known Issues

- bad UX to start initial embedding computation on startup and not user trigger since it can hog resources
- button, command+keyboard shortcut, and menu item(?) to show/hide the panel
- - if this is really common, would be cool to abstract that into a lib or smt
- when switching between notes quickly, it's not clear list is outdated for a few secs
  - need to abort the computation onSelectionChange
- visually distinguish linked notes from unlinked notes in results (since they are obv already known/accounted for by user)
- setting to exclude specified notebooks from being included (borrow more code from Note Graph UI plugin)
- optimize similarity calculations, and when they happen, for more responsive UI
- save USE model locally, so it needn't be downloaded every time the plugin loads
- fix potential edge case of note embedding unsyncing from note content
- could recompute onSync, compromising between onSelect and onChange
- change list UI to look identical to default joplin note list

## Future Work

- viz note similarities in 2d or 3d - tfjs supports this to some extent
- optionally include note's tags in the embeddings (test how this changes results per note in practice)
- - see gensim impl here: https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e
- compare results of USE lite with mobileBERT
- compare results of USE lite with topic extraction + keyword search
- summarize each note via some other LM, and show summary blurb in results list, to help user know what's in each similar note
  - maybe instead allow on-hover previews of the note? or on-hover/button summarizations/outline
- wonder what results would be if we calculated similarity of multiple selected notes
  (using onSelectedNoteIds event). could average the selected note embeddings

## Questions

- not sure if I should be doing preprocessing on the data before creating the embedding?
  - some documentation says I should (sentencepiece), but the official examples of USE in tfjs don't. maybe tf vs tfjs thing?
  - top2vec does preprocessing before it calls USE to create the embedding...
  - on l2 normalization: https://stackoverflow.com/questions/32276391/feature-normalization-advantage-of-l2-normalization
- USE embeddings are deterministic, right? don't need to recalc embeddings like top2vec suggests? (maybe that's only for top2vec algo, not USE within top2vec?)

## Licenses

This plugin's source code is MIT licensed. Tensorflow,js is Apache 2.0 (their license is included in the .jpl)

---

# Joplin Plugin

This is a template to create a new Joplin plugin.

The main two files you will want to look at are:

- `/src/index.ts`, which contains the entry point for the plugin source code.
- `/src/manifest.json`, which is the plugin manifest. It contains information such as the plugin a name, version, etc.

## Building the plugin

The plugin is built using Webpack, which creates the compiled code in `/dist`. A JPL archive will also be created at the root, which can use to distribute the plugin.

To build the plugin, simply run `npm run dist`.

The project is setup to use TypeScript, although you can change the configuration to use plain JavaScript.

## Updating the plugin framework

To update the plugin framework, run `npm run update`.

In general this command tries to do the right thing - in particular it's going to merge the changes in package.json and .gitignore instead of overwriting. It will also leave "/src" as well as README.md untouched.

The file that may cause problem is "webpack.config.js" because it's going to be overwritten. For that reason, if you want to change it, consider creating a separate JavaScript file and include it in webpack.config.js. That way, when you update, you only have to restore the line that include your file.
