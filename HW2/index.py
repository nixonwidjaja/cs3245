#!/usr/bin/python3
import getopt
import math
import nltk
import os
import pickle
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

    def merge(self, other):
        self.plist = [i.value if isinstance(i, Posting) else i for i in self.plist]
        other = [i.value if isinstance(i, Posting) else i for i in other]
        self.plist.extend(other)
        self.finalize()


class Indexer:
    def __init__(self, out_dict, out_postings) -> None:
        self.stemmer = nltk.stem.PorterStemmer()
        self.dictionary = {}
        self.word_to_pointer_dict = {}
        self.out_dict = out_dict
        self.out_postings = out_postings

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
        with open(self.out_postings, "wb"):
            pass
        with open(self.out_postings, "ab") as f:
            for word, pl in self.dictionary.items():
                pointer = f.tell()
                data = pickle.dumps(pl)
                f.write(data)
                self.word_to_pointer_dict[word] = (pointer, len(data))
        with open(self.out_dict, "wb") as f:
            pickle.dump(self.word_to_pointer_dict, f)

    def load(self):
        with open(self.out_dict, "rb") as f:
            self.word_to_pointer_dict = pickle.load(f)

    def get_posting_list(self, word: str):
        if len(self.word_to_pointer_dict) == 0:
            self.load()
        word = self.stemmer.stem(word.lower(), to_lowercase=True)
        with open(self.out_postings, "rb") as f:
            if word not in self.word_to_pointer_dict:
                return []
            pointer, length = self.word_to_pointer_dict[word]
            f.seek(pointer)
            data = f.read(length)
        return pickle.loads(data)


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print("indexing...")
    # This is an empty method
    # Pls implement your code in below
    indexer = Indexer(out_dict, out_postings)
    for _, _, files in os.walk(in_dir):
        for file in files:
            indexer.index(os.path.join(in_dir, file), int(file))
    indexer.finalize()


def test_get_posting_lists(out_dict, out_postings):
    print("test get posting lists...")
    indexer = Indexer(out_dict, out_postings)
    for word in ["billion", "u.s.", "dollar", "week", ",", "mln"]:
        print(word)
        print(indexer.get_posting_list(word))


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
test_get_posting_lists(output_file_dictionary, output_file_postings)
# python3 index.py -i ./reuters/small-training -d dictionary.txt -p postings.txt
