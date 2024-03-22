#!/usr/bin/python3
from index import Indexer

# These imports are necessary for Pickle.load
# Python needs to know what classes are being deserialized into so we need
# to load the classes into memory for Pickle to work
from index import WordToPointerEntry, PostingList, Posting
import math
import nltk
import sys
import getopt


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"
    )


def preprocess(text: str, stemmer: nltk.stem.PorterStemmer):
    # words = text.strip().lower().split()
    # return [stemmer.stem(w, to_lowercase=True) for w in words]
    words = text.strip().lower()
    ret = []
    for sent in nltk.sent_tokenize(words):
        for word in nltk.word_tokenize(sent):
            ret.append(stemmer.stem(word, to_lowercase=True))
    return ret


def search(words: list[str], indexer: Indexer):
    N = indexer.get_N()
    scores = {}
    count = {}
    for word in words:
        if word not in count:
            count[word] = 0
        count[word] += 1
    for word in words:
        pl = indexer.get_posting_list(word)
        df = indexer.get_df(word)
        for posting in pl.plist:
            docId, tfd = posting.docId, posting.tf
            if docId not in scores:
                scores[docId] = 0
            wtd = 1 + math.log10(tfd)
            wtq = (1 + math.log10(count[word])) * math.log10(N / df)
            scores[docId] += wtd * wtq
    for d in scores.keys():
        scores[d] /= indexer.get_doc_lengths(d)
    items = list(scores.items())
    items.sort(key=lambda x: x[0])
    items.sort(key=lambda x: x[1], reverse=True)
    items = [str(i[0]) for i in items]
    return " ".join(items[:10])


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print("running search on the queries...")
    # This is an empty method
    # Pls implement your code in below

    indexer = Indexer(dict_file, postings_file)
    indexer.load()
    stemmer = nltk.stem.PorterStemmer()

    with open(results_file, "w") as outf, open(queries_file, "r") as inf:
        queries = inf.readlines()
        for query in queries:
            query = query.strip()
            print(f"OG Query: {query}")
            words = preprocess(query, stemmer)
            print(f"After transform: {words}")
            result = search(words, indexer)
            outf.write(result + "\n")
            break


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "d:p:q:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == "-d":
        dictionary_file = a
    elif o == "-p":
        postings_file = a
    elif o == "-q":
        file_of_queries = a
    elif o == "-o":
        file_of_output = a
    else:
        assert False, "unhandled option"

if (
    dictionary_file == None
    or postings_file == None
    or file_of_queries == None
    or file_of_output == None
):
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
