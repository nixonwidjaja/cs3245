#!/usr/bin/python3
import re
import math
import nltk
import sys
import getopt
import heapq

from collections import defaultdict
from index import Indexer
# These imports are necessary for pickle to work
from index import Posting, PostingList, WordToPointerEntry


class Score:
    """A custom score object with a custom defined
    __lt__ to meet the ordering specified"""
    def __init__(self, score, docId):
        self.score = score
        self.docId = docId
        
    def __repr__(self):
        return f"(Score={self.score},docId={self.docId})"
        
    def __lt__(self, other):
        # Sort by score then by docId
        return (self.score > other.score) or (self.score == other.score and self.docId < other.docId)

def preprocess_query(query: str, stemmer: nltk.stem.StemmerI) -> list[str]:
    """Preprocess the query using nltk"""
    query = query.strip()
    ret = []
    for sent in nltk.sent_tokenize(query):
        for word in nltk.word_tokenize(sent):
            ret.append(stemmer.stem(word.lower(), to_lowercase=True))
    return ret


def get_term_freq(query: list[str]) -> dict[str, int]:
    """Given a query, compute the mapping of term to occurrences"""
    term_counts = defaultdict(int)
    for t in query:
        term_counts[t] += 1
    return term_counts
        

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")
    
    
def get_tf(term, term_counts) -> float:
    """Get the tf of a term given the Indexer"""
    # If num_terms[term] = 0, something has gone horribly wrong
    tf = 1 + math.log(term_counts[term], 10)
    return tf


def get_idf(term: str, indexer: Indexer) -> float:
    """Get the idf of a term given the Indexer"""
    N = indexer.get_N()
    df = indexer.get_df(term)
    if df == 0:
        return 0
    return math.log(N / df)
    

def compute_w_t_q(term: str, 
                  indexer: Indexer,
                  term_counts: dict[str, int]):
    """Compute the weight of a term given the Indexer and the query counts"""
    # If num_terms[term] = 0, something has gone horribly wrong
    tf = get_tf(term, term_counts)
    idf = get_idf(term, indexer)
    w_t_q = tf * idf
    return w_t_q

def compute_w_t_d(tf_t_d):
    """Given the tf_t_d, compute w_t_d. Currently uses the math.log(_, 10)"""
    # If num_terms[term] = 0, something has gone horribly wrong
    tf = 1 + math.log(tf_t_d, 10)
    # The other term is times 1, so identity
    return tf

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
    scores = {}
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
        for posting in pl:
            d = posting.docId
            if d not in scores:
                scores[d] = Score(0, d)
            # Alr precomputed
            w_t_d = posting.tf
            scores[d].score += w_t_d * w_t_q
    for docId in scores.keys():
        scores[docId].score = scores[docId].score / length[docId]
    # Invert the docId since we are interested in
    # ascending docId as tiebreakers.
    # The heapq is largest, so inverting will make the
    # smaller ones go first 
    heap = [(s.score, -s.docId) for s in scores.values()]
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
    indexer = Indexer(dict_file, postings_file)
    with open(queries_file, "r") as qf, open(results_file, "w") as wf:
        import tqdm
        for i, line in enumerate(tqdm.tqdm(qf.readlines())):
            line = line.strip()
            docIds = search(line, indexer, K=K)
            wf.write(" ".join(list(map(str, docIds))) + "\n")


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
