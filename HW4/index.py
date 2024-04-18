#!/usr/bin/python3
import getopt
import math
import os
import re
import sys
from collections import Counter

import nltk

from preprocessor import Preprocessor


def usage():
    print(
        "usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file"
    )


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print("indexing...")
    doc_ids = [int(x) for x in os.listdir(in_dir)]
    doc_norm_lengths: dict[int, float] = {}
    inverted_index: dict[str, list[tuple[int, int]]] = {}

    for doc_id in sorted(doc_ids):
        filepath = os.path.join(in_dir, str(doc_id))
        tf_dict = Counter(Preprocessor.file_to_token_stream(filepath))
        doc_norm_lengths[doc_id] = math.sqrt(
            sum((1 + math.log10(tf)) ** 2 for tf in tf_dict.values())
        )

        for term, tf in tf_dict.items():
            postings_list = inverted_index.get(term, [])
            postings_list.append((doc_id, tf))
            inverted_index[term] = postings_list

    with (
        open(out_dict, "w") as odict,
        open(out_postings, "w") as opost,
    ):
        start_offset = 0
        for term, postings_list in inverted_index.items():
            # Write the DF, offset and size for each term into dictionary file.
            opost.write(f'{" ".join([f"({doc_id},{tf})" for doc_id, tf in postings_list])}\n')
            end_offset = opost.tell()
            size = end_offset - start_offset

            # Write the DF, offset and size for each term into dictionary file.
            df = len(postings_list)
            odict.write(f"{term} {df} {start_offset} {size}\n")
            start_offset = end_offset
        odict.write("\n")

        # Append the normalized lengths to dictionary file.
        for docid, length in doc_norm_lengths.items():
            odict.write(f"{docid} {length}\n")


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
