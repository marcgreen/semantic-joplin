# Semantically Similar Notes

## WARNING - will crash if you have large (megabyte+) notes (ignoring attachments)

anecdotally: I had a 1mb note, several 100kb notes, and ~800 notes smaller than that, totalling 6mb. the model hung for me when trying to embed all of these in one go. excluding the 1mb note, I could embed everything else in one go just fine.

so, not sure what the actual note size limit is for the model. I tried to have it save intermediate embedding progress to disk, so even if the model hangs, you can restart joplin and it should resume where it left off. but I wasn't able to get this working as I'd expect. it seems like the db INSERTs aren't being committed early enough. you can see the .sqlite file grow in size between each crash, but not as quickly as it should be...

please let me know your system specs and note stats (how big are your largest notes, how many notes do you have, what is your total note size in MB excl. attachments?) if the model crashes for you.

## Summary

The intent of this plugin is to help the joplin user find notes they have that might be relevant to the one they are editing. We plan to filter out linked notes despite relevancy, as an explicit linking implies the user is already aware of the relevance. This filtering is not yet implemented, however.

Disclaimer: This plugin was written during a 2-week hackathon, without prior javascript or tensorflow experience, and the code quality reflects this. PRs more than welcome :)

### On First Use

On first startup, this plugin will calculate embeddings for all of your notes. This can take some minutes (less than 2min for my 800 notes totalling several hundred KBs on my desktop computer). It saves these embeddings to disk and loads from there on subsequent startups.

If it hangs/crashes during the initial embeddings computation, try restarting joplin. The plugin should resume where it left off before crashing. it might take many many restarts though, since the code to save these embeddings doesn't work as intended...


## Semanticness

We use tensorflow.js's [Universal Sentence Encoder Lite](https://github.com/tensorflow/tfjs-models/tree/master/universal-sentence-encoder) to encode each note into a 512 dimension vector, called an embedding. It's called a 'sentence encoder', but it seems to work for longer strings of text, too (eg, see doc2vec and top2vec). (The limit seems to be somewhere between 100 and ~1000 kilobytes of text, though this limit might be a bug in tfjs.) We use the dot product of two embeddings to determine similarity.

It might be the case that better results could be achieved with a different model (eg mobileBERT, or BERT), or perhaps an entirely different approach, like one that does a simple keyword search instead (assuming a way to determine the right keywords to use, perhaps via topic extraction).

### Quality of Results

Sometimes the results of the semantic similarity computation are confusing or questionable. Consider that the model is comparing what it thinks the meaning of the words you're using in the selected note against all other notes. So there are three variables at play: the words used in the selected note, the words used in the note being compared against, and the model's understanding of our language.

Testing the model against my own corpus of notes, I am satisified enough with its performance. But I would certainly be interested in trying out different models (and different approaches), like the ones mentioned in Future Work below.

## Caveats/Limitations

- requires internet connection to download the USE model every time. would like to save it locally after first load
- need to recreate the model if # of notes significantly changes
  - would it help to recreate model over time, too, after many things are removed/added?
- no support for attachments yet
- not sure if I should be doing preprocessing on the data before creating the embedding.
  - top2vec does preprocessing before it calls USE to create the embedding...
  - on l2 normalization: https://stackoverflow.com/questions/32276391/feature-normalization-advantage-of-l2-normalization

## Future Work / Bugfixes

- start initial embedding computation only on user trigger, not automatically
- need to use event queue to re-embed notes that were sync'd from other devices...
  - save event cursor to disk so we can ensure we don't miss anything
- when switching between notes quickly, it's not clear list is outdated for a few secs
  - need to abort the computation onSelectionChange
- remove linked notes from results (since they are obv already known/accounted for by user)
- setting to exclude specified notebooks from being included (borrow more code from Note Graph UI plugin)
- optimize similarity calculations, and when they happen, for more responsive UI
- save USE model locally, so it needn't be downloaded every time the plugin loads
- fix potential edge case of note embedding unsyncing from note content
- note list will still show notes that were deleted until next launch
- names of new notes created won't be visible in list until next launch
  - both this and showing deleted notes can be resolved by moving part of
    getAllNoteEmbeddings() into updateSimilarNoteList()
- should recompute onSync, compromising between onSelect and onChange
- button to show/hide the web panel?
- change UI to look identical to default joplin note list
- viz note similarities in 2d or 3d - tfjs supports this to some extent
- optionally include note's tags in the embeddings (test how this changes results per note in practice)
- - see gensim impl here: https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e
- compare results of USE lite with mobileBERT
- compare results of USE lite with topic extraction + keyword search
- summarize each note via some other LM, and show summary blurb in results list, to help user know what's in each similar note
  - maybe instead allow on-hover previews of the note? or on-hover/button summarizations/outline
- wonder what results would be if we calculated similarity of multiple selected notes
  (using onSelectedNoteIds event). could average the selected note embeddings

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
