#!/usr/bin/python3
import math
import nltk
import sys
import getopt
import heapq
import time

from collections import defaultdict, Counter
from index import Indexer
# These imports are necessary for pickle to work
from index import Posting, PostingList, WordToPointerEntry


def preprocess_query(query: str, stemmer: nltk.stem.StemmerI) -> list[str]:
    """
    Use techniques from NLTK such as word and sent tokenize as well as
    stemming to preprocess a given text. Applies case folding last.
    """
    query = query.strip()
    ret = []
    sentences = nltk.sent_tokenize(query)
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            ret.append(stemmer.stem(word).lower())
    return ret


def get_term_freq(query: list[str]) -> dict[str, int]:
    """Given a query, compute the mapping of term to occurrences"""
    return Counter(query)
        

def usage():
    """Prints usage for search.py"""
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")
    
    
def get_tf(term, term_counts) -> float:
    """Get the tf of a query term given the Indexer"""
    # If num_terms[term] = 0, something has gone horribly wrong
    tf = 1 + math.log(term_counts[term], 10)
    return tf


def get_idf(term: str, indexer: Indexer) -> float:
    """Get the idf of a query term given the Indexer"""
    N = indexer.get_N()
    df = indexer.get_df(term)
    if df == 0:
        return 0
    return math.log(N / df)
    

def compute_query_vector(terms: list[str], indexer: Indexer, term_counts):
    """Compute the query-normalised query vector for a query given indexer"""
    length = 0
    query_vector = {}
    for term in terms:
        tf = get_tf(term, term_counts)
        idf = get_idf(term, indexer)
        weight = tf * idf
        query_vector[term] = weight
    length = math.hypot(*list(query_vector.values()))
    if length > 0:
        for term in terms:
            query_vector[term] /= length
    return query_vector
    

def search(query, indexer: Indexer, K=10):
    """Compute relevant documents using the cosine score redux algorithm"""
    query = preprocess_query(query, indexer.stemmer)
    # We store scores as (score, docId)
    # heapify uses the first attribute, so we want to 'sort' by score
    # then afterwards, we retain the docId as the return value
    scores = defaultdict(float)
    length = indexer.get_doc_length()   
    # Obtain the term counts to avoid doing repeated work
    term_counts = get_term_freq(query)
    # Compute the query vector separately so that we can
    # normalize it separately as well
    query = list(set(query))
    query_vector = compute_query_vector(query, indexer, term_counts)
    for term in set(query):
        w_t_q = query_vector[term]
        pl = indexer.get_posting_list(term)
        if pl is None:
            continue
        for posting in pl.plist:
            d = posting.docId
            # Alr precomputed
            w_t_d = posting.tf
            scores[d] += w_t_d * w_t_q
    for docId in scores.keys():
        scores[docId] = scores[docId] / length[docId]
    # Invert the docId since we are interested in
    # ascending docId as tiebreakers.
    # The heapq is largest, so inverting will make the
    # smaller ones go first
    heap = [(score, -docId) for docId, score in scores.items()] 
    heap = heapq.nlargest(K, heap)
    results = [-item[1] for item in heap]
    return results


def run_search(dict_file, postings_file, queries_file, results_file, K=10):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    K = top K documents to retrieve
    """
    print('running search on the queries...')
    start_time = time.time()
    indexer = Indexer(dict_file, postings_file)
    with open(queries_file, "r") as qf, open(results_file, "w") as wf:
        for line in qf.readlines():
            line = line.strip()
            docIds = search(line, indexer, K=K)
            wf.write(" ".join(list(map(str, docIds))) + "\n")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}")


if __name__ == "__main__":
    dictionary_file = postings_file = file_of_queries = output_file_of_results = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-d':
            dictionary_file  = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"

    if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
        usage()
        sys.exit(2)
    run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
    # with Profile() as profile:
    #     Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()