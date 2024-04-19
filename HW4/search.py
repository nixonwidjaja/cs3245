#!/usr/bin/python3
import getopt
import math
import sys
import time
from collections import defaultdict
from functools import cmp_to_key

import nltk
from indexer import Indexer
from query_parser import QueryParser
from scorer import Scorer


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results"
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


def run_search(
    dict_path: str,
    postings_path: str,
    queries_path: str,
    out_results_path: str,
) -> None:
    print(f'Searching for the query "{queries_path}" ...')
    start_time = time.time()

    # Load the input query/relevant Doc-IDs.
    with open(queries_path, "r") as f:
        query, *relevant_doc_ids = (line.rstrip("\n") for line in f.readlines())

    # Compute scores.
    with Indexer(dict_path, postings_path) as indexer:
        query_tokens = QueryParser.get_query_tokens(query)
        scorer = Scorer(indexer)
        scorer.init_term_weights(query_tokens)
        scores = scorer.get_doc_scores()

    # Sort scores (tie break by doc-ID).
    elements = [(-score, doc_id) for doc_id, score in scores.items()]
    elements.sort(key=cmp_to_key(compare_tuples))
    output_doc_ids = [str(doc_id) for _, doc_id in elements]

    # Write results to output file.
    with open(out_results_path, "w") as f:
        f.write(" ".join(output_doc_ids))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.1f}s")

    for doc_id in relevant_doc_ids:
        print(f"{doc_id:10}: Rank {output_doc_ids.index(doc_id)}")


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
