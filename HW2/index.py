#!/usr/bin/python3
import getopt
import math
import nltk
import os
import pickle
import re
import sys


class Posting:
    def __init__(self, value) -> None:
        self.value = value
        self.skip = None

    def __repr__(self):
        return f"(value = {self.value} skip = {self.skip})"


class PostingsList:
    def __init__(self) -> None:
        self.plist = []

    def __repr__(self):
        return str(self.plist)

    def append(self, value):
        self.plist.append(value)

    def finalize(self):
        self.plist = sorted(list(set(self.plist)))
        skips = round(math.sqrt(len(self.plist)))
        step = len(self.plist) // skips
        self.plist = [Posting(val) for val in self.plist]
        for i in range(0, len(self.plist), step):
            if i + step < len(self.plist):
                self.plist[i].skip = i + step


class Indexer:
    def __init__(self, out_dict) -> None:
        self.stemmer = nltk.stem.PorterStemmer()
        self.dictionary = {}
        self.out_dict = out_dict

    def index(self, file: str, doc_id: int):
        with open(file, "r") as f:
            text = f.read().lower()
        words = nltk.word_tokenize(text)
        singles = set([self.stemmer.stem(w, to_lowercase=True) for w in words])
        for s in singles:
            if s not in self.dictionary:
                self.dictionary[s] = PostingsList()
            self.dictionary[s].append(doc_id)

    def finalize(self):
        for s in self.dictionary:
            self.dictionary[s].finalize()
        with open(self.out_dict, "wb") as f:
            pickle.dump(self.dictionary, f)
        with open(self.out_dict, "rb") as f:
            print(pickle.load(f))


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print("indexing...")
    # This is an empty method
    # Pls implement your code in below
    indexer = Indexer(out_dict)
    for _, _, files in os.walk(in_dir):
        for file in files:
            indexer.index(os.path.join(in_dir, file), int(file))
    indexer.finalize()


input_directory = output_file_dictionary = output_file_postings = None


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -i directory-of-documents -d dictionary-file -p postings-file"
    )


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

if (
    input_directory == None
    or output_file_postings == None
    or output_file_dictionary == None
):
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
# python3 index.py -i ./reuters/small-training -d dictionary.txt -p postings.txt
