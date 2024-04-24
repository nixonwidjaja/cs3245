#!/usr/bin/python3
import getopt
import math
import pickle
import sys
import time

import nltk
from dataset import Dataset
from indexer import vb_encode
from tqdm.autonotebook import tqdm


def usage():
    filename = sys.argv[0]
    print(
        f"usage: {filename}"
        + " -i dataset-file"
        + " -d dictionary-file"
        + " -p postings-file"
    )  # fmt:skip


def build_index(
    dataset_path: str,
    out_dict_path: str,
    out_postings_path: str,
    use_compression: bool = False,
) -> None:
    print("indexing...")
    start_time = time.time()

    inverted_index: dict[str, list[tuple[int, float]]] = {}
    doc_vectors: dict[int, dict[str, float]] = {}
    for doc_id, tokens in tqdm(
        Dataset.get_tokenized_content_stream(dataset_path), total=Dataset.NUM_DOCUMENTS
    ):
        tf_dict = nltk.FreqDist(tokens)
        norm_len = math.sqrt(sum((1 + math.log10(tf)) ** 2 for tf in tf_dict.values()))
        vector: dict[str, float] = {}

        for term, tf in tf_dict.items():
            doc_weight = (1 + math.log10(tf)) / norm_len
            vector[term] = doc_weight

            postings_list = inverted_index.get(term, [])
            postings_list.append((doc_id, doc_weight))
            inverted_index[term] = postings_list

        doc_vectors[doc_id] = vector
    if use_compression:
        # Do gap and variable encoding on the doc id for each posting list
        for term, postings_list in inverted_index.items():
            gaps = [postings_list[0][0]]
            doc_weights = [postings_list[0][1]]
            for i in range(1, len(postings_list)):
                doc_id, doc_weight = postings_list[i]
                gaps.append(doc_id - postings_list[i - 1][0])
                doc_weights.append(doc_weight)
            vb_gaps = vb_encode(gaps)
            inverted_index[term] = (vb_gaps, doc_weights)

    term_metadata: dict[str, tuple[int, int, int]] = {}
    doc_metadata: dict[int, tuple[int, int]] = {}
    with open(out_postings_path, "wb") as post_f:
        start_offset = 0
        for term, postings_list in inverted_index.items():
            post_f.write(pickle.dumps(postings_list))
            end_offset = post_f.tell()
            size = end_offset - start_offset
            if use_compression:
                df = len(postings_list[1])
            else:
                df = len(postings_list)
            term_metadata[term] = (df, start_offset, size)

            start_offset = end_offset

        for doc_id, vector in doc_vectors.items():
            post_f.write(pickle.dumps(vector))
            end_offset = post_f.tell()
            size = end_offset - start_offset

            doc_metadata[doc_id] = (start_offset, size)
            start_offset = end_offset

    with open(out_dict_path, "wb") as dict_f:
        pickle.dump((term_metadata, doc_metadata), dict_f)

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
