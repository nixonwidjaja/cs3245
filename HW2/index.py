#!/usr/bin/python3

from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from heapq import merge
import getopt
import math
import nltk
import os
import pickle
import sys
import time


class Posting:
    def __init__(self, value) -> None:
        self.value = value
        self.skip = None

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __repr__(self):
        return f"(value = {self.value} skip = {self.skip})"

    def has_skip(self):
        return self.skip is not None

    def __hash__(self):
        return hash(self.value)


class PostingsList:
    def __init__(self) -> None:
        self.plist = []

    def __repr__(self):
        return str(self.plist)

    def __len__(self):
        return len(self.plist)

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

    def append(self, value: Posting):
        self.plist.append(value)

    def finalize(self):
        self.plist = sorted(list(set(self.plist)), key=lambda p: p.value)
        skips = round(math.sqrt(len(self.plist)))
        step = len(self.plist) // skips
        for i in range(0, len(self.plist), step):
            if i + step < len(self.plist):
                self.plist[i].skip = i + step

    def merge(self, other):
        self.plist.extend(other.plist)
        if len(self.plist) == 0:
            return
        for i in self.plist:
            i.skip = None
        self.finalize()


@dataclass
class WordToPointerEntry:
    # Where in the file, the posting list begins
    pointer: int
    # How many additional bytes to read
    pointer_offset: int
    # How many items in the posting list
    size: int

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


# The term used to represent the list of all doc ids
UNIVERSE = ""


"""
Indexer:
    take in parameters
    
"""

class DocumentStream:
    """Provides a convenient generator over the token stream"""
    def __init__(self, dir):
        self.dir = dir
        
    def token_stream(self):
       ...
    

@dataclass
class DocumentStreamToken:
    term: str
    docId: int
    
        
def tokenize_document(docId, path, processing_fn):
    with open(path, "r") as inf:
        text = inf.read()
        # convenient list set hax
        text = list(set(processing_fn(text)))
        for token in text:
            yield DocumentStreamToken(token, docId)
            

def tokenize_collection(dir, processing_fn):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        # Assume that doc id is name of file
        docId = int(file)
        for token in tokenize_document(docId, path, processing_fn):
            yield token


class Indexer:
    def __init__(self, out_dict, out_postings, block_dir="block", block_size=500000) -> None:
        """
        The posting files contains the serialized version of the posting lists.
        The dictionary file contains a serialized version of the term to pointer in the postings list.
        When we run SPIMI, we will create a block file of each term to list of docIds.
        Default to 0.5MB for the block size
        """
        self.stemmer = nltk.stem.PorterStemmer()
        self.dictionary = {}
        self.word_to_pointer_dict = {}
        self.out_dict = out_dict
        self.out_postings = out_postings
        self.universe = PostingsList()
        self.block_dir = block_dir
        self.block_size = block_size
        # You should not be doing this here, else we can't use the Indexer
        # Maybe make another method
        # if os.path.exists(out_dict):
        #     os.remove(out_dict)
        # if os.path.exists(out_postings):
        #     os.remove(out_postings)
        
    def index_collection(self, collection_dir):
        token_stream = tokenize_collection(collection_dir, processing_fn=self.preprocess_text)
        print("SPIMI Inverting...")
        num_blocks = self.spimi_invert(token_stream)
        print("SPIMI Inverting done.")
        # self.merge_blocks(num_blocks)
        # Now to block-way merging
    
    def spimi_invert(self, token_stream):
        def flush_dictionary():
            nonlocal dictionary, curr_block_number
            sorted_dict = self.sort_dictionary(dictionary)
            self.flush_block(curr_block_number, sorted_dict)
            dictionary = defaultdict(list)
            curr_block_number += 1
            
        import shutil
        if os.path.exists(self.block_dir):
            shutil.rmtree(self.block_dir)
        os.mkdir(self.block_dir)
        
        # At the end of it, curr block number will indicate how many blocks were written to disk
        curr_block_number = 0
        # in-memory dictionary that will be flushed to disk
        dictionary = defaultdict(list)
        
        for token in token_stream:
            docId, term = token.docId, token.term
            dictionary[term].append(docId)
            if sys.getsizeof(dictionary) > self.block_size:
                flush_dictionary()
        # Make sure to handle the last remaining dictionary if it has entries
        if dictionary:
            flush_dictionary()
        return curr_block_number
           
    @staticmethod 
    def sort_dictionary(dictionary: dict):
        """Dictionary is mapping from unsorted terms to unsorted list of docIds.
        We want to sort it into a sorted dictionary by terms and by docIds as seen in slide 22 of the lecture slides.
        """
        # Use an OrderedDict because it remembers insertion order.
        new_dict = OrderedDict()
        # Sort terms in alphabetical order and insert in alphabetical order
        terms = list(sorted(dictionary.keys()))
        for term in terms:
            docIds = sorted(dictionary[term])
            new_dict[term] = docIds
        return new_dict
    
    
    def get_block_filename(self, block_id) -> str:
        """Helper method to obtain block filename given block id"""
        return os.path.join(self.block_dir, f"block_{block_id}.txt")
    
    def flush_block(self, block_id: int, dict: dict):
        with open(self.get_block_filename(block_id), "w") as inf:
            for term, docIds in dict.items():
                inf.write(f"{term}: {docIds}\n")

    def merge_blocks(self, num_blocks: int):
        """Maintain num_block pointers to each block file and advance line by line"""
        # Let n = num_blocks
        # Create n file I/O objects
        block_files = []
        block_lines = []
        for block_id in range(num_blocks):
            block_files.append(open(self.get_block_filename(block_id), "r"))
        # with open("")
        # In here, we will initialise the dictionary of term to pointer as well as the posting list itself
        while True:
            # Read one line from each block_file
            for i in range(num_blocks):
                block_lines[i] = block_files[i].readline()
            for block_file in block_files:
                ...
            # Clear each term at a time
            
            ...
        
        # Remember to close them all
        for block_file in block_files:
            block_file.close()

    def get_memory_size(self):
        return sys.getsizeof(self.dictionary)

    def preprocess_text(self, text: str) -> list[str]:
        """
        Preprocessing done for indexing as well as searching.
        Move to own static method to standardize across.
        """
        text = text.lower()
        words = nltk.word_tokenize(text)
        singles = [self.stemmer.stem(w, to_lowercase=True) for w in words]
        return singles

    def index(self, file: str, doc_id: int):
        with open(file, "r") as f:
            text = f.read().lower()
        singles = set(self.preprocess_text(text))
        for s in singles:
            if s not in self.dictionary:
                self.dictionary[s] = PostingsList()
            self.dictionary[s].append(Posting(value=doc_id))
        # Add the doc id to the UNIVERSE to maintain list of all docs
        self.universe.append(Posting(value=doc_id))

    def finalize(self):
        for s in self.dictionary:
            self.dictionary[s].finalize()
        with open(self.out_postings, "wb") as f:
            for word, pl in self.dictionary.items():
                pointer = f.tell()
                data = pickle.dumps(pl)
                f.write(data)
                self.word_to_pointer_dict[word] = WordToPointerEntry(
                    pointer, len(data), len(pl)
                )
        with open(self.out_dict, "wb") as f:
            pickle.dump(self.word_to_pointer_dict, f)

    def spimi(self):
        print("spimi")
        temp_file = "temp.txt"
        # Move the current postings file to a temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(self.out_postings):
            os.rename(self.out_postings, temp_file)
        if os.path.exists(self.out_dict):
            self.load()
        # Initialize new word to pointer dict to store new pointers
        new_word_to_pointer_dict = {}
        # Collect a set of all words (in memory and temp_file)
        words = set(self.word_to_pointer_dict.keys())
        words = words.union(set(self.dictionary.keys()))
        # Write to out_postings the updated index
        with open(self.out_postings, "wb") as f:
            for word in words:
                # Get old posting list (in temp_file)
                old_list = self.get_posting_list(word, temp_file)
                # Get new posting list (in memory)
                new_list = self.dictionary.get(word, PostingsList())
                # Merge old and new posting lists
                old_list.merge(new_list)
                # Get current pointer in out_postings
                pointer = f.tell()
                # Write byte data to out_postings and store the pointer to dict
                data = pickle.dumps(old_list)
                f.write(data)
                new_word_to_pointer_dict[word] = WordToPointerEntry(
                    pointer, len(data), len(old_list)
                )
        # Reset dicts for next iteration
        self.dictionary = {}
        # Write word_to_pointer_dict to out_dict file
        with open(self.out_dict, "wb") as f:
            pickle.dump(new_word_to_pointer_dict, f)
        # Remove temp_file
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def load(self):
        with open(self.out_dict, "rb") as f:
            self.word_to_pointer_dict = pickle.load(f)

    def get_posting_list(self, word: str, filename=None) -> PostingsList:
        filename = self.out_postings if filename is None else filename
        word = self.stemmer.stem(word.lower(), to_lowercase=True)
        if word not in self.word_to_pointer_dict or not os.path.exists(filename):
            return PostingsList()
        with open(filename, "rb") as f:
            entry = self.word_to_pointer_dict[word]
            f.seek(entry.pointer)
            data = f.read(entry.pointer_offset)
        return pickle.loads(data)

    def get_full_postings(self):
        ans = {}
        self.load()
        for word in self.word_to_pointer_dict:
            ans[word] = str(self.get_posting_list(word))
        return ans


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print("indexing...")
    # This is an empty method
    # Pls implement your code in below
    indexer = Indexer(out_dict, out_postings)
    indexer.index_collection(in_dir)
    # Try with hax
    # indexer = Indexer(out_dict, out_postings)
    # for _, _, files in os.walk(in_dir):
    #     for file in files:
    #         indexer.index(os.path.join(in_dir, file), int(file))
    # indexer.finalize()
    # A = indexer.get_full_postings()
    # print(sys.getsizeof(A))
    # MEMORY_LIMIT = int(1e6)
    
    # # Try with SPIMI
    # indexer = Indexer(out_dict, out_postings)
    # for _, _, files in os.walk(in_dir):
    #     for file in files:
    #         indexer.index(os.path.join(in_dir, file), int(file))
    #         if indexer.get_memory_size() > MEMORY_LIMIT:
    #             indexer.spimi()
    #             indexer.dictionary = {}
    # indexer.spimi()
    # B = indexer.get_full_postings()
    # print(sys.getsizeof(B))
    
    # # Compare
    
    # print(A == B)
    # for k in A:
    #     if k not in B:
    #         print(k)
    #     elif A[k] != B[k]:
    #         print(len(A[k]))
    #         print(len(B[k]))
    #         print()
    # for k in B:
    #     if k not in A:
    #         print(k)
    #     elif A[k] != B[k]:
    #         print(len(A[k]))
    #         print(len(B[k]))
    #         print()


def test_get_posting_lists(out_dict, out_postings):
    print("test get posting lists...")
    indexer = Indexer(out_dict, out_postings)
    indexer.load()
    print(indexer.get_posting_list(""))
    # for word in ["billion", "u.s.", "dollar", "week", ",", "mln"]:
    #     print(word)
    #     print(type(indexer.get_posting_list(word)))
    #     print(indexer.word_to_pointer_dict[word])
    #     break


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -i directory-of-documents -d dictionary-file -p postings-file"
    )


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

    build_index(input_directory, output_file_dictionary, output_file_postings)
    # test_get_posting_lists(output_file_dictionary, output_file_postings)
    # python3 index.py -i ./reuters/small-training -d dictionary.txt -p postings.txt
