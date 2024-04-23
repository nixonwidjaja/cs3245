#!/usr/bin/python3
import getopt
import itertools
import math
import re
import sys
import time
from collections import defaultdict

import phrase_search
from indexer import Indexer
from preprocessor import Preprocessor
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
) -> None:
    print(f'Searching for the query "{queries_path}" ...')
    start_time = time.time()

    # Load the input query/relevant Doc-IDs.
    with open(queries_path, "r") as f:
        query, *relevant_doc_ids = (line.rstrip("\n") for line in f.readlines())

    # Compute scores.
    with Indexer(dict_path, postings_path) as indexer:
        if is_bool_query := '"' in query or "AND" in query:
            phrases: list[str] = [x.replace('"', "") for x in re.split(r"\s+AND\s+", query)]
            query_tokens: list[list[str]] = []
            for phrase in phrases:
                query_tokens.append(list(Preprocessor.to_token_stream(phrase)))
            tf_list = phrase_search.solve_query(indexer, query_tokens)

            scores: dict[int, float] = defaultdict(lambda: 0.0)
            for doc_id, tfs in tf_list:
                scores[doc_id] += sum(1 + math.log10(tf) for tf in tfs)
                elements = [(-score, doc_id) for doc_id, score in scores.items()]
                elements.sort()
                output_doc_ids = [str(doc_id) for _, doc_id in elements]
        else:
            synonyms_list: list[set[str]] = [
                QueryParser.get_synonyms(token) for token in Preprocessor.to_token_stream(query)
            ]
            query_combinations = list(itertools.product(*synonyms_list))

            result: dict[str, None] = {}
            for permute in phrase_search.search_permutes(len(synonyms_list)):
                scores: dict[int, float] = defaultdict(lambda: 0.0)
                for combi in query_combinations:
                    query_tokens = phrase_search.apply_permutation(list(combi), permute)
                    tf_list = phrase_search.solve_query(indexer, query_tokens)
                    for doc_id, tfs in tf_list:
                        score = 0.0
                        for tokens, tf in zip(query_tokens, tfs):
                            df = min(indexer.get_df(t) for t in tokens if indexer.get_df(t) > 0)
                            if df != 0:
                                score += 1 + math.log10(tf)
                        scores[doc_id] = max(scores[doc_id], score)
                if len(result) > 1000:
                    break

                elements = [(-score, doc_id) for doc_id, score in scores.items()]
                elements.sort()
                for _, doc_id in elements:
                    result[str(doc_id)] = None
            output_doc_ids = list(result.keys())
            print()

    # Sort scores (tie break by doc-ID).

    # Write results to output file.
    with open(out_results_path, "w") as f:
        f.write(" ".join(output_doc_ids))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.1f}s")

    for doc_id in relevant_doc_ids:
        try:
            print(f"{doc_id:10}: Rank {output_doc_ids.index(doc_id) + 1}")
        except:
            print(f"{doc_id:10}: Rank -1")


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
