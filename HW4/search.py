#!/usr/bin/python3
import getopt
import heapq
import math
import re
import sys
import time
from collections import Counter, defaultdict
from functools import cmp_to_key
from operator import index

import nltk

from indexer import Indexer
from preprocessor import Preprocessor


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"
    )


def compare_tuples(v1: tuple[float, int], v2: tuple[float, int]) -> int:
    """Comparison function for my list of (-score, doc_id), to account for
    floating-point errors in the score.

    This will treat score floats as equal if they're very close.
    """
    float1, int1 = v1
    float2, int2 = v2

    if math.isclose(float1, float2, rel_tol=1e-9):
        return int1 - int2
    elif float1 < float2:
        return -1
    else:
        return 1


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print("running search on the queries...")
    start_time = time.time()

    with (
        Indexer(dict_file, postings_file) as indexer,
        open(queries_file, "r") as oqueries,
        open(results_file, "w") as oresults,
    ):
        is_first_line: bool = True
        N: int = indexer.num_docs

        for query in oqueries:
            query = query.rstrip("\n")
            tf_dict = Counter(Preprocessor.to_token_stream(query))

            # Compute un-normalized scores first.
            query_norm_length = 0
            scores: dict[int, float] = defaultdict(lambda: 0.0)
            for term, tf in tf_dict.items():
                df, postings_list = indexer.get_term_data(term)

                # Ignore terms that don't appear in any docs.
                if df == 0:
                    continue

                query_weight = (1 + math.log10(tf)) * math.log10(N / df)  # Log-TF-IDF for query.
                query_norm_length += query_weight**2

                for doc_id, tf in postings_list:
                    doc_weight = 1 + math.log10(tf)  # Log-TF (w/o IDF) for documents.
                    scores[doc_id] += doc_weight * query_weight

            query_norm_length = math.sqrt(query_norm_length)

            # Do cosine normalization on the scores.
            for doc_id in scores.keys():
                doc_norm_length = indexer.doc_norm_lengths[doc_id]
                scores[doc_id] /= doc_norm_length * query_norm_length

            # Get the top 10 highest scores (tie break by doc-ID).
            heap_items = [(-score, doc_id) for doc_id, score in scores.items()]
            top_items = heapq.nsmallest(10, heap_items, key=cmp_to_key(compare_tuples))
            top_doc_ids = [str(doc_id) for _, doc_id in top_items]

            padding = "" if is_first_line else "\n"
            oresults.write(padding + " ".join(top_doc_ids))
            is_first_line = False

    end_time = time.time()
    print(f"Execution time: {end_time - start_time}s")


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
