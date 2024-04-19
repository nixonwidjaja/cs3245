#!/usr/bin/python3
import getopt
import math
import pickle
import sys
import time

import nltk
from dataset import Dataset
from tqdm.autonotebook import tqdm


def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")


def build_index(dataset_path: str, out_dict_path: str, out_postings_path: str) -> None:
    print("indexing...")
    start_time = time.time()

    inverted_index: dict[str, list[tuple[int, float]]] = {}
    for doc_id, tokens in tqdm(
        Dataset.get_tokenized_content_stream(dataset_path), total=Dataset.NUM_DOCUMENTS
    ):
        tf_dict = nltk.FreqDist(tokens)
        norm_len = math.sqrt(sum((1 + math.log10(tf)) ** 2 for tf in tf_dict.values()))

        for term, tf in tf_dict.items():
            postings_list = inverted_index.get(term, [])
            postings_list.append((doc_id, (1 + math.log10(tf)) / norm_len))
            inverted_index[term] = postings_list

    term_metadata: dict[str, tuple[int, int, int]] = {}
    with open(out_postings_path, "wb") as post_f:
        start_offset = 0
        for term, postings_list in inverted_index.items():
            post_f.write(pickle.dumps(postings_list))
            end_offset = post_f.tell()
            size = end_offset - start_offset

            df = len(postings_list)
            term_metadata[term] = (df, start_offset, size)

            start_offset = end_offset

    with open(out_dict_path, "wb") as dict_f:
        pickle.dump((term_metadata,), dict_f)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.1f}s")


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:d:p:")
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == "-i":  # input directory
        input_directory = a
    elif o == "-d":  # dictionary file
        output_file_dictionary = a
    elif o == "-p":  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
