#!/usr/bin/python3
import getopt
import math
import pickle
import sys
import time
import gzip
import shutil

from struct import pack, unpack

import nltk
from dataset import Dataset
from tqdm.autonotebook import tqdm


def usage():
    filename = sys.argv[0]
    print(
        f"usage: {filename}"
        + " -i dataset-file"
        + " -d dictionary-file"
        + " -p postings-file"
    )  # fmt:skip
    
    
def __encode(number):
    b = []
    while True:
        b.insert(0, number % 128)
        if number < 128:
            break
        number = number // 128
    b[-1] += 128
    return pack('%dB' % len(b), *b)


def gap_encode(numbers) -> list[int]:
    gaps = [numbers[0]]
    for i in range(1, len(numbers)):
        gaps.append(numbers[i] - numbers[i-1])
    return gaps


def vb_encode(numbers):
    bytes_list = []
    for number in numbers:
        bytes_list.append(__encode(number))
    return b"".join(bytes_list)


def vb_decode(bytestream):
    n = 0
    numbers = []
    bytestream = unpack('%dB' % len(bytestream), bytestream)
    for byte in bytestream:
        if byte < 128:
            n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0
    return numbers


def build_index(dataset_path: str, out_dict_path: str, out_postings_path: str) -> None:
    print("indexing...")
    start_time = time.time()

    inverted_index: dict[str, dict[int, list[int]]] = {}

    for doc_id, tokens in tqdm(
        Dataset.get_tokenized_content_stream(dataset_path), total=Dataset.NUM_DOCUMENTS
    ):
        # we now need to store the df separately before doing the vbe postings
        encountered_docs: dict[str, list[int]] = {}
        # Positional indexing token -> pos id -> posting list
        pos_indices_dict: dict[str, dict[int, list[int]]] = {}
        for pos, token in enumerate(tokens):
            if token not in pos_indices_dict:
                pos_indices_dict[token] = {}
                
            if pos not in pos_indices_dict[token]:
                pos_indices_dict[token][pos] = []

            pos_indices_dict[token][pos].append(doc_id)

        for term, pos_indices in pos_indices_dict.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            if term not in encountered_docs:
                encountered_docs[term] = []
            encountered_docs[term].append(doc_id)
            for pos, postings in pos_indices.items():
                if pos not in inverted_index[term]:
                    inverted_index[term][pos] = []
                for doc in postings:
                    inverted_index[term][pos].append(doc)
    
    # Now do gap and variable byte encoding for each mini-posting list
    for term, pos_indices in inverted_index.items():
        for pos, postings in pos_indices.items():
            gap_postings = gap_encode(postings)
            vbe_postings = vb_encode(gap_postings)
            pos_indices[pos] = vbe_postings

    term_metadata: dict[str, tuple[int, int, int]] = {}
    doc_metadata: dict[int, tuple[int, int]] = {}
    with open(out_postings_path, "wb") as post_f:
        start_offset = 0
        for term, postings_list in inverted_index.items():
            post_f.write(pickle.dumps(postings_list))
            end_offset = post_f.tell()
            size = end_offset - start_offset

            df = len(encountered_docs[term])
            term_metadata[term] = (df, start_offset, size)

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
