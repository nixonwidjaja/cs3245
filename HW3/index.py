#!/usr/bin/python3
import math
import nltk
import sys
import getopt
import os
import pickle
from dataclasses import dataclass
import time

class Posting:
    """
    Posting abstraction which represents the docId and the term frequency
    """

    def __init__(self, docId):
        self.docId = docId
        self.tf = 1

    def __repr__(self) -> str:
        return f"(doc={self.docId}, tf={self.tf})"


class PostingList:
    """
    PostingList contains list of postings as in the lecture material.
    """

    def __init__(self):
        self.plist = []

    def append(self, value: Posting):
        self.plist.append(value)

    def sort(self, key: str):
        if key == "docid":
            self.plist = sorted(self.plist, key=lambda p: p.docId)
        elif key == "tf":
            self.plist = sorted(self.plist, key=lambda p: p.tf)
        else:
            raise ValueError(f"Unsupported key type {key}, should be 'docid' or 'tf'")

    def __len__(self):
        return len(self.plist)

    def __repr__(self):
        return str(self.plist)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self):
            raise StopIteration
        x = self.plist[self.i]
        self.i += 1
        return x

    def __getitem__(self, i):
        return self.plist[i]
    
    


@dataclass
class WordToPointerEntry:
    """
    Dataclass used for serialisation of dict to dictionary.txt
    """
    # Where in the file, the PostingList begins
    pointer: int
    # How many additional bytes to read
    pointer_offset: int
    # Doc frequency for the word
    df: int

    def __getstate__(self) -> object:
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


class Indexer:
    """Class used to index the collection of documents into a dictionary and postings file."""
    def __init__(self, out_dict, out_postings, sortkey: str = "docid"):
        self.out_dict = out_dict
        self.out_postings = out_postings
        self.stemmer = nltk.stem.PorterStemmer()
        self.dictionary = {}
        self.word_to_ptr_dict = {}
        self.doc_lengths = {}
        if sortkey not in ["docid", "tf"]:
            raise ValueError("sortkey should either be 'docid' or 'tf'")
        self.sortkey = sortkey

    def preprocess_text(self, text: str) -> list[str]:
        """
        Use techniques from NLTK such as word and sent tokenize as well as
        stemming to preprocess a given text
        """
        sentences = nltk.sent_tokenize(text)
        words = []
        for sent in sentences:
            words.extend(nltk.word_tokenize(sent))
        # In general, case folding should be the last step since the case
        # itself is commonly used to indicate some linguistic information 
        # such as proper nouns/start of a sentence - Zhao Jin
        singles = [self.stemmer.stem(w).lower() for w in words]
        return singles
    

    def index_collection(self, collection):
        """
        In-memory indexing of the collection
        We first write to our in-memory dict then we write to the file
        """
        max_doc_id = 0
        for file in os.listdir(collection):
            path = os.path.join(collection, file)
            docId = int(file)
            with open(path, "r") as f:
                text = f.read()
            singles = self.preprocess_text(text)
            # Because we want to store the term frequency, we will compute
            # a sub-dictionary of all the Postings first and then add it to
            # the main dictionary afterwards
            sub_dict = {}
            for s in singles:
                if s not in sub_dict:
                    sub_dict[s] = Posting(docId)
                else:
                    sub_dict[s].tf += 1
            # As seen in piazza, normalisation of the 
            # doc length should be based on the 
            # log(tf) + 1 values instead of raw freq
            # Pre-compute the tf weights
            all_tfs = []
            for k, p in sub_dict.items():
                precomputed_tf = 1 + math.log10(p.tf)
                p.tf = precomputed_tf
                all_tfs.append(precomputed_tf)
            # For some reason, this two lines are actually not equivalent!
            # self.doc_lengths[docId] = math.hypot(*all_tfs)
            self.doc_lengths[docId] = math.sqrt(sum(tf ** 2 for tf in all_tfs))
            # Now we have the docId and tf for all the terms, add it to dict
            for single, posting in sub_dict.items():
                if single not in self.dictionary:
                    self.dictionary[single] = PostingList()
                self.dictionary[single].append(posting)

        with open(self.out_postings, "wb") as outf:
            for single, pl in self.dictionary.items():
                pl.sort(key=self.sortkey)
                pointer = outf.tell()
                data = pickle.dumps(pl)
                outf.write(data)
                self.word_to_ptr_dict[single] = WordToPointerEntry(
                    pointer, len(data), len(pl)
                )
        with open(self.out_dict, "wb") as outf:
            pickle.dump(self.word_to_ptr_dict, outf)
        with open("lengths.txt", "wb") as outf:
            pickle.dump(self.doc_lengths, outf)

    def load(self):
        """
        Loads the dict file into word_to_ptr_dict for retrieval
        """
        with open(self.out_dict, "rb") as f:
            self.word_to_ptr_dict = pickle.load(f)
        with open("lengths.txt", "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_df(self, word):
        """get the doc frequency for a word in the collection"""
        if word not in self.word_to_ptr_dict:
            return 0
        return self.word_to_ptr_dict[word].df
    
    def get_N(self):
        """get the total number of documents in the collection"""
        if not self.doc_lengths:
            self.load()
        return len(self.doc_lengths.keys())

    def get_posting_list(self, word, filename=None):
        """Use low level file operation to read in the
        Pickle serialized file, deserialize into PostingList"""
        if not self.word_to_ptr_dict:
            self.load()
        filename = self.out_postings if filename is None else filename
        if word not in self.word_to_ptr_dict.keys():
            return None
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as inf:
            entry = self.word_to_ptr_dict[word]
            inf.seek(entry.pointer)
            data = inf.read(entry.pointer_offset)
        return pickle.loads(data)
    
    def get_doc_lengths(self, term):
        """Get the doc lengths vector for a term"""
        if not self.doc_lengths:
            self.load()
        if term not in self.doc_lengths:
            return self.doc_lengths[term]
        return 1
    
    def get_doc_length(self):
        """Retrieve the document length vector, loading from disk if necessary"""
        if not self.doc_lengths:
            self.load()
        return self.doc_lengths


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -i directory-of-documents -d dictionary-file -p postings-file"
    )


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print("indexing...")
    start_time = time.time()
    indexer = Indexer(out_dict, out_postings)
    indexer.index_collection(in_dir)
    end_time = time.time()
    print(f"Took {end_time - start_time} seconds to index")
    return indexer


def test_get_posting_lists(out_dict, out_postings):
    """Test getting posting lists"""
    indexer = Indexer(out_dict, out_postings)
    indexer.load()
    for word in ["employe"]:
        pl = indexer.get_posting_list(word)
        print(pl)


if __name__ == "__main__":
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

    if (
        input_directory == None
        or output_file_postings == None
        or output_file_dictionary == None
    ):
        usage()
        sys.exit(2)

    indexer = build_index(input_directory, output_file_dictionary, output_file_postings)
    # print(indexer.get_posting_list("chu"))
    # print(indexer.get_posting_list("housing,"))
    # test_get_posting_lists(output_file_dictionary, output_file_postings)
