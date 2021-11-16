import os
import re
import sys
import glob
import collections

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
    min_cluster_size = 5

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
                        # todo could try using tokenizer of ^
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
                            # oh the function get_documents_topics exists lol
                            #
                            # use search_documents_by_documents() to impl main idea,
                            # and filter out ones already linked to it.
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

    #import pprint
    #pprint.pprint(topic_nums)
    #pprint.pprint(topic_score)
    
    for i, _ in enumerate(topic_nums):
        output.append("\n")
        output.append(f"title: {titles[i]}")
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

# i bet many notes could fall under one or more category. or maybe the goal is to find broad enough buckets so that isn't the case?. can i force hierarchical modeling?

# see research Qs above

# is it possible to impl top2vec in js given existing libs, and specifically
# as a joplin plugin? i'm personally more interested in the reesarch Qs, but
# feel some obligate to engineer something. if joplin plugin isn't realistic
# in the time frame, then what is an acceptable alternative UX?
# - could server REST api via python BE and just have joplin plugin be the FE for it
# - - this way I could still integrate most similar docs to given doc into UI
# - - wonder if params above affect document-document similarity search at all? LEFT OFF HERE
