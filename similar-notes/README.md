# Semantically Similar Notes

The intent of this plugin is to help the joplin user find notes they have that might be relevant to the one they are editing. We plan to filter out linked notes despite relevancy, as an explicit linking implies the user is already aware of the relevance. This filtering is not yet implemented, however.

Disclaimer: This plugin was written during a 2-week hackathon, without prior javascript or tensorflow familiarity. I'm grateful it seems to even work like one would expect. PRs more than welcome :)

## Semanticness

We use tensorflow.js's [Universal Sentence Encoder Lite](https://github.com/tensorflow/tfjs-models/tree/master/universal-sentence-encoder) to encode each note into a 512 dimension vector, called an embedding. It's called a 'sentence encoder', but it seems to work for longer strings of text, too (eg, see doc2vec and top2vec). We use the dot product of two embeddings to determine similarity.

It might be the case that better results could be achieved with a different model (eg mobileBERT, or BERT), or perhaps an entirely different approach, like one that does a simple keyword search instead (assuming a way to determine the right keywords to use).

## Caveats/Limitations

- need to recreate the model if # of notes significantly changes
  - would it help to recreate model over time, too, after many things are removed/added?
- not sure how many notes would be needed to benefit from a different LM (eg BERT)
- not sure how well this works for smaller # of notes. development was done with a corpus of 800 notes (6mb text)
- no support for attachments yet

## Future Work

- setting to exclude specified notebooks from being included (borrow more code from Note Graph UI plugin)
- add option to remove linked notes from results (since they are obv already known/accounted for by user)
- viz note similarities in 2d or 3d
- optionally include note's tags in the embeddings (test how this changes results per note in practice)
- - see gensim impl here: https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e

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
