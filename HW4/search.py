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
        N = indexer.num_docs
        query_tokens = QueryParser.get_query_tokens(query)
        tf_dict = nltk.FreqDist(query_tokens)

        # Compute un-normalized scores first.
        scores: dict[int, float] = defaultdict(lambda: 0.0)
        for term, tf in tf_dict.items():
            df, postings_list = indexer.get_term_data(term)

            # Ignore terms that don't appear in any docs.
            if df == 0:
                continue

            query_weight = (1 + math.log10(tf)) * math.log10(N / df)  # Log-TF-IDF for query.

            for doc_id, tf in postings_list:
                doc_weight = 1 + math.log10(tf)  # Log-TF (w/o IDF) for documents.
                scores[doc_id] += doc_weight * query_weight

        # Do cosine normalization on the scores.
        for doc_id in scores.keys():
            doc_norm_length = indexer.doc_norm_lengths[doc_id]
            scores[doc_id] /= doc_norm_length

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
