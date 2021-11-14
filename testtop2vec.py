import os
import re
import sys
import glob
from top2vec import Top2Vec

# ignoring resources for now, and all other type_s but 1

# takes directory of exported joplin notes
def raw_to_corpus(export_dir):
    corpus = []
    # look at every md file
    # - the bottom of each note has key/values starting with id, ending with type_
    # - skip notes that aren't type_:1
    # - for type_:1 notes, strip their metadata to not distract top2vec
    
    rgx = re.compile(r"""(([^\n]+).+)              # grab the title, and the data
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
                #print(title)
            else:
                print(text)
                sys.exit("err, regex for note wrong")
    # return notes as list of strings
    return corpus

def main():
    # todo try loading saved model, if exception, create it
    #model = Top2Vec.load("filename")
    
    export_dir = '/mnt/s/Documents/semantic-joplin/joplin-kg-nov2021/'
    documents = raw_to_corpus(export_dir)
    print(len(documents))

    # default hdbscan_args:
    # 'min_cluster_size': 15,
    # 'metric': 'euclidean',
    # 'cluster_selection_method': 'eom'
    model = Top2Vec(documents, embedding_model='universal-sentence-encoder',
                    # speed="deep-learn",
                    workers=8,
                    min_count=1,
                    hdbscan_args={
                        # todo left off exploring 17 topics found with these settings, and tweaknig the settings to see if i can find better groupings
                        # can i viz the topics somehow?
                        # - LEFT OFF HERE: word cloud pick is one way
                        # - project into lower dimension? would like to see all the documents and the topic centroids
                        'min_cluster_size': 15,
                        'min_samples': 1,
                        # cluster_selection_epsilon
                    }
    )
    num_topics = model.get_num_topics()
    topic_words, word_scores, topic_nums = model.get_topics()
    print(num_topics)
    print(topic_words)
    model.save("kg-model-1") # once i ahve a good model, optionally load it above (todo)
    
    return 0 # no error

if __name__ == '__main__':
    sys.exit(main())
