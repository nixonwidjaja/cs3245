#!/usr/bin/python3
import getopt
import sys
import time

from indexer import Indexer
from query_parser import QueryParser
from scorer import Scorer


def usage():
    filename = sys.argv[0]
    print(
        f"usage: {filename}"
        + " -d dictionary-file"
        + " -p postings-file"
        + " -q query-file"
        + " -o output-file-of-results"
    )


def run_search(
    dict_path: str,
    postings_path: str,
    queries_path: str,
    out_results_path: str,
    use_compression: bool = False,
) -> None:
    print(f'Searching for the query "{queries_path}" ...')
    start_time = time.time()

    # Load the input query/relevant Doc-IDs.
    with open(queries_path, "r") as f:
        query, *relevant_doc_ids = (line.rstrip("\n") for line in f.readlines())

    # Compute scores.
    with Indexer(dict_path, postings_path, use_compression) as indexer:
        query_tokens = QueryParser.get_query_tokens(query)
        scorer = Scorer(indexer)
        scorer.init_term_weights(query_tokens)
        scorer.apply_pseudo_relevance_feedback(
            alpha=0.9,
            n_relevant=5,
            beta=0.1,
            n_irrelevant=100,
            gamma=0.1,
        )
        scores = scorer.get_doc_scores()

    # Sort scores (tie break by doc-ID).
    elements = [(-score, doc_id) for doc_id, score in scores.items()]
    elements.sort()
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
