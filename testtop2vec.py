import os
import re
import sys
import glob
import pprint
import collections
import umap.plot
import matplotlib.pyplot as plt

from top2vec import Top2Vec

# ignoring resources for now, and all other type_s but 1

# takes directory of exported joplin notes
def raw_to_corpus(export_dir):
    corpus = []
    titles = []
    # look at every md file
    # - the bottom of each note has key/values starting with id, ending with type_
    # - skip notes that aren't type_:1
    # - for type_:1 notes, strip their metadata to not distract top2vec
    
    rgx = re.compile(r"""(([^\n]*).+)?             # grab the title, and the data
                         ^id:.+?type_:\ (\d)\Z""", # this is the meta data
                     re.MULTILINE | re.DOTALL | re.VERBOSE)
    for filename in glob.glob(os.path.join(export_dir, '*.md')):
        with open(filename, 'r') as f:
            text = f.read()
            match = rgx.match(text)
            if match:
                notetype = int(match.group(3))
                if notetype != 1:
                    continue # next note
                data = match.group(1)
                corpus.append(data)
                title = match.group(2)
                titles.append(title)
                #print(title)
            else:
                print(text)
                sys.exit("err, regex for note wrong")
    # return notes as list of strings
    return titles, corpus

def main():
    # default 50 – Ignores all words with total frequency lower than this.
    # For smaller corpora a smaller min_count will be necessary.
    min_count = 1

    # smallest size grouping that you wish to consider a cluster
    min_cluster_size = 2
    # how conservative you want you clustering to be. The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, and clusters will be restricted to progressively more dense areas
    min_samples = 1
    
    model_name = f"kg-model-{min_count}-{min_cluster_size}-{min_samples}"
    export_dir = '/mnt/s/Documents/semantic-joplin/joplin-kg-nov2021/'
    # load or train a model
    try:
        model_path = f"models/{model_name}"
        model = Top2Vec.load(model_path)
        do_save_model = False

        # need to do this for the titles. todo save the processed notes to disk
        titles, documents = raw_to_corpus(export_dir)
    except Exception as e:
        print(f"Exception {e}")
        print("training new model")

        titles, documents = raw_to_corpus(export_dir)

        # need unique document ids, so need unique document titles
        dups = [item for item, count in collections.Counter(titles).items() if count > 1]
        if len(dups) > 0:
            print(dups)
            print("error need unique document titles, see above duplicated titles")
            exit()
            
        print(len(documents))
        # default hdbscan_args:
        # 'min_cluster_size': 15,
        # 'metric': 'euclidean',
        # 'cluster_selection_method': 'eom'
        model = Top2Vec(documents, document_ids = titles,
                        embedding_model='universal-sentence-encoder',
                        use_embedding_model_tokenizer=True, # is this better or worse?
                        workers=8,
                        min_count=min_count,
                        hdbscan_args={
                            # exploring topics found with these settings,
                            # tweaking settings to see if i can find better groupings.
                            # can i viz the topics somehow?
                            # - word cloud pic is one way
                            # - project into lower dimension? would like to see all the
                            #   documents and the topic centroids
                            #   - yes, whitepaper shows example of doing this TODO
                            # - pick some document titles at random from each topic?
                            #   or look at all of the titles in each topic?
                            # experimentally determine which min_count, min_cluster_size,
                            # min_samples values are the best for my data
                            #
                            # maybe don't focus on topics themselves, but on similar-
                            # document-recommendation capability for each document?
                            # well, need them at least looking coherent i guess?
                            # - todo use average document-topic similarity to judge fit?
                            # - either adjust the 3 major params OR 1 2 1 hiera reduce
                            #   until maximized avg document-topic similarity?
                            # - research Q
                            #
                            # how important is it that the top X number of similar words
                            # to a topic uniquely define that topic? could minimize X
                            # - todo
                            # - maybe 1 2 1 hiera reduce for X = 3, 10 and see results?
                            # - how should we know what X should be? should X be 1?
                            # - or do we not hiera reduce, and instead look at top Y
                            #   topics for a given document to uniquely describe it?
                            #   - this reminds me of gunk vs junk in mereology
                            # - research Q
                            #
                            # oh, maybe we 1 2 1 hiera reduce until word scores are
                            # high enough? but how high? maximize the average?
                            # - research Q
                            # 
                            # are these hierarchical by default? no.
                            # - todo: try calling hierarchical_topic_reduction on 1 2 1
                            #         with varying num_topics. can we xy plot something?
                            #         and/or viz the centroid document changing as the
                            #         hierarchy builds up?
                            #
                            # could use negative keywords/doc_ids to let user refine
                            # their search as they see irrelevant docs popping up?
                            #
                            #
                            #
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                        # cluster_selection_epsilon
                        }
        )
        model.save(model_path)

    
    # we now have our model
    output = [] # this is tee'd to stdout and file below
    num_dims = model.topic_vectors.shape[1]
    num_topics = model.get_num_topics()

    output.append(f"{num_topics} {num_dims}-dimension topics in this model")
    #topic_sizes, topic_nums = model.get_topic_sizes()
    #topic_words, word_scores, topic_nums = model.get_topics()
    #for i, topic_num in enumerate(topic_num):
    #    output.append(f"\ntopic #{topic_num} (size {topic_sizes[i]})")
    #    a = [f"{wrd} ({scr:.2%})" for scr, wrd in zip(word_scores[i], topic_words[i])]
    #    output.append(",".join(a))
    #    output.append(topic_nums[i])
    #    # print(model.generate_topic_wordcloud(topic_nums[-1]))
    #    # todo, want to save plot instead of showing it, bc i'm in wsl2
    #    # - use plt.savefig("matplotlib.png") inside top2vec func ^

    # is this different from search_documents_by_topics?
    topic_nums, topic_score, topic_words, word_scores = \
        model.get_documents_topics(titles) # todo try num_topics = 5 or 10?

    #pprint.pprint(topic_nums)
    #pprint.pprint(topic_score)

    # todo get the vec output of top2vec, i think embedding with 512 dimensions with USE?
    # - input into umap object to conver to 2 dim, and save plot to disk
    #
    # The vector dimension should be the same as the vectors in the topic_vectors variable. (i.e. model.topic_vectors.shape[1])
    # get umap model from model
    # passing reduced topics to them as the label to color code by
    # but if i don't call hierarchical_topic_reduction, does this return the normal
    # doc_top, or null? lemme just use doc_top to see if that works
    points = umap.plot.points(umap_model, labels=model.doc_top)
    #fig = plt.figure()
    #fig.add_subplot(points)
    #fig 
    #fig.savefig(f"output/topics-{model_name}.png")
    
    pprint.pprint(points)
    #chart = umap.plot.points(umap_model, labels=model.doc_top_reduced)
    # is the data continuous or categorical? i think categorical. but docs can
    # belong to more than one topic, but that still seems categorical.
    # save the chart
    
    
    for i in range(len(topic_nums)):
        # for each document, find semantically similar docs
        # - consider filtering out docs that are already linked.
        # - can supply list of doc_ids to search against
        # - can supply negative doc_ids for dissimilarity
        _, scores, ids = model.search_documents_by_documents([titles[i]], 5)
        output.append("\n")
        output.append(f"title: {titles[i]}")
        output.append(f"similar documents:")
        for i2 in range(len(ids)):
            output.append(f"  {scores[i2]:.1%}: {ids[i2]}")

        output.append(f"topic #{topic_nums[i]} w/ similarity {topic_score[i]:.2%}%")
        output.append(f"words similar to the topic of this document:\n")
        a = [f"{wrd} ({scr:.2%})" for scr, wrd in zip(word_scores[i], topic_words[i])]

        output.append(",".join(a))
        #output.append(topic_nums[i])
        pass
        
    
    print("\n".join(output))
    with open(f"output/output-{model_name}", "w") as fh:
        fh.writelines(output)
    
    return 0 # no error

if __name__ == '__main__':
    sys.exit(main())

## thoughts

# https is the first word of topics 3 and 4 for model 10 30 1. wonder if they are so common because of all the links I have? if so, would like to strip them out (http too)
# - i think this just shows that the documents in the topics with these as similar words just have a lot of links in them
# - can i add this as a stop word so the tokenizer would delete it from the corpus?

# i bet many notes could fall under one or more category. or maybe the goal is to find broad enough buckets so that isn't the case?. can i force hierarchical modeling?

# see research Qs above

# is it possible to impl top2vec in js given existing libs, and specifically
# as a joplin plugin? if joplin plugin isn't realistic in the time frame,
# then what is an acceptable alternative UX?
# - could server REST api via python BE and just have joplin plugin be the FE for it
# - - this way I could still integrate most similar docs to given doc into UI
# - - wonder if params above affect document-document similarity search at all?
# - - - doesn't look like they do
# - tensorflow.js with node backend looks promising
#
# can I find a relationship between:
# - hierarchical_topic_reduction in these models, and
# - the model of hierarchical complexity
# want to viz the hierarchy, can use get_topic_hierarchy
#
# interactive plotting sounds really fun and like an effective way to explore the
# embedding: https://umap-learn.readthedocs.io/en/latest/plotting.html#interactive-plotting-and-hover-tools
# - TODO create interactive notebook, where user can train new model, or select a model to viz or output info about it
