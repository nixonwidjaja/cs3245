#!/usr/bin/python3
import getopt
import math
import sys
from collections import Counter

from dataset import Dataset
from tqdm.autonotebook import tqdm


def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")


def build_index(dataset_path: str, out_dict_path: str, out_postings_path: str) -> None:
    print("indexing...")
    doc_norm_lengths: dict[int, float] = {}
    inverted_index: dict[str, list[tuple[int, int]]] = {}

    for doc_id, tokens in tqdm(Dataset.get_tokenized_content_list(dataset_path)):
        tf_dict = Counter(tokens)
        doc_norm_lengths[doc_id] = math.sqrt(
            sum((1 + math.log10(tf)) ** 2 for tf in tf_dict.values())
        )

        for term, tf in tf_dict.items():
            postings_list = inverted_index.get(term, [])
            postings_list.append((doc_id, tf))
            inverted_index[term] = postings_list

    with (
        open(out_dict_path, "w") as dict_f,
        open(out_postings_path, "w") as post_f,
    ):
        start_offset = 0
        for term, postings_list in inverted_index.items():
            # Write the DF, offset and size for each term into dictionary file.
            post_f.write(f'{" ".join([f"({doc_id},{tf})" for doc_id, tf in postings_list])}\n')
            end_offset = post_f.tell()
            size = end_offset - start_offset

            # Write the DF, offset and size for each term into dictionary file.
            df = len(postings_list)
            dict_f.write(f"{term} {df} {start_offset} {size}\n")
            start_offset = end_offset
        dict_f.write("\n")

        # Append the normalized lengths to dictionary file.
        for docid, length in doc_norm_lengths.items():
            dict_f.write(f"{docid} {length}\n")


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
