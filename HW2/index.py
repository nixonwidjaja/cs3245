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


class Posting:
    def __init__(self, value) -> None:
        self.value: int = value
        self.skip: int = None

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

    def add_skip_pointers(self):
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
        self.add_skip_pointers()


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
UNIVERSE = "-+@\/.,./"  # I am choosing random terms to make this unique


"""
Indexer:
    take in parameters
    
"""


class DocumentStream:
    """Provides a convenient generator over the token stream"""

    def __init__(self, dir):
        self.dir = dir

    def token_stream(self): ...


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


def tokenize_collection(dir, processing_fn, debug=True):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        # Assume that doc id is name of file
        docId = int(file)
        # Create debug dir if no exist
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        if debug:
            with open(f"{debug_dir}/debug_{file}.txt", "w") as outf:
                outf.write(str(list(tokenize_document(docId, path, processing_fn))))
        for token in tokenize_document(docId, path, processing_fn):
            yield token


class Indexer:
    def __init__(
        self, out_dict, out_postings, block_dir="block", block_size=500000, use_binary=True
    ) -> None:
        """
        The posting files contains the serialized version of the posting lists.
        The dictionary file contains a serialized version of the term to pointer in the postings list.
        When we run SPIMI, we will create a block file of each term to list of docIds.
        Default to 0.5MB for the block size
        Note that use_Binary=False is purely for debugging and will not work with loading at the moment cos deserialization would
        be slightly harder to write.
        """
        self.stemmer = nltk.stem.PorterStemmer()
        self.dictionary = {}
        self.word_to_pointer_dict = {}
        self.out_dict = out_dict
        self.out_postings = out_postings
        self.universe = PostingsList()
        self.block_dir = block_dir
        self.block_size = block_size
        self.use_binary = use_binary

    def index_collection(self, collection_dir):
        token_stream = tokenize_collection(
            collection_dir, processing_fn=self.preprocess_text
        )
        print("SPIMI Inverting...")
        num_blocks = self.spimi_invert(token_stream)
        print("SPIMI Inverting done!")
        print("Merging blocks...")
        self.merge_blocks(num_blocks, collection_dir)
        print("Blocks merged!")

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

    def merge_blocks(self, num_blocks: int, collection_dir: str):
        """Maintain num_block pointers to each block file and advance line by line"""

        def can_still_process(block_lines):
            # Stop processing when all are none
            result = not all(line is None for line in block_lines)
            return result

        # Let n = num_blocks
        # Create n file I/O objects
        block_files = []
        for block_id in range(num_blocks):
            block_files.append(open(self.get_block_filename(block_id), "r"))
        block_lines = [block_files[i].readline() for i in range(num_blocks)]
        # with open("")
        # In here, we will initialise the dictionary of term to pointer as well as the posting list itself
        self.word_to_pointer_dict = {}
        # As mentioned in the specifications, we will write the postings list with the skip pointers since the index is already constructed here
        # We assume that the dictionary to the WPE fits into memory
        mode = "wb" if self.use_binary else "w"
        with open(self.out_postings, mode) as out_pf:
            while can_still_process(block_lines):
                # Extract the terms
                terms = []
                for i in range(num_blocks):
                    if block_lines[i]:
                        # We do a "".join(...[:-1]) to deal with weird terms that may contain : itself
                        terms.append("".join(block_lines[i].split(":")[:-1]))
                    else:
                        terms.append(None)
                # Choose the alphabetically smallest one
                # Need to handle for empty term, sort and push None to the end
                sorted_terms = sorted(terms, key=lambda x: (x is None, x))
                # This is guaranteed not to be None
                smallest_term = sorted_terms[0]
                smallest_term_doc_ids = []
                for i in range(num_blocks):
                    if terms[i] == smallest_term:
                        # Convert list str representation from text file to python list of ints
                        doc_ids = [
                            int(s)
                            for s in block_lines[i]
                            .split(":")[-1]
                            .strip()
                            .replace("[", "")
                            .replace("]", "")
                            .split(", ")
                        ]
                        smallest_term_doc_ids.append(doc_ids)
                        # Advance the file reader, if there are no more lines to read
                        line = block_files[i].readline()
                        if "\n" not in line and line == "":
                            block_lines[i] = None
                        else:
                            block_lines[i] = line
                # Make use of the battle tested, lazy loading heapq.merge to do it for us!
                # merge here assumes that each of the input is sorted (which it is) and does not pull the data all into memory
                # We need to unwrap the iterables here
                doc_ids = list(merge(*smallest_term_doc_ids))
                postings = [Posting(v) for v in doc_ids]
                posting_list = PostingsList()
                for posting in postings:
                    posting_list.append(posting)
                posting_list.add_skip_pointers()

                # Write to out postings
                out_pf_ptr = out_pf.tell()
                pl_data = pickle.dumps(posting_list) if self.use_binary else str(posting_list)
                out_pf.write(pl_data)

                # print to str for verification
                # out_pf.write(str(posting_list))

                # Write to out dict
                wpe = WordToPointerEntry(out_pf_ptr, len(pl_data), len(posting_list))
                self.word_to_pointer_dict[smallest_term] = wpe
            # We are done with the SPIMI merge here but we are going to add the Universe entry
            universe_posting_list = PostingsList()
            for file in os.listdir(collection_dir):
                posting = Posting(value=int(file))
                universe_posting_list.append(posting)
            universe_posting_list.add_skip_pointers()
            out_pf_ptr = out_pf.tell()
            pl_data = pickle.dumps(universe_posting_list) if self.use_binary else str(universe_posting_list)
            out_pf.write(pl_data)
            wpe = WordToPointerEntry(
                out_pf_ptr, len(pl_data), len(universe_posting_list)
            )
            self.word_to_pointer_dict[UNIVERSE] = wpe
        # Remember to close them all
        for block_file in block_files:
            block_file.close()
        with open(self.out_dict, mode) as out_df:
            if self.use_binary:
                pickle.dump(self.word_to_pointer_dict, out_df)
            else:
                out_df.write(str(self.word_to_pointer_dict))
        print("Done indexing!")

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
            self.dictionary[s].add_skip_pointers()
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
    print(f"indexing {in_dir} to dictionary file {out_dict} and postings file {out_postings}")
    indexer = Indexer(out_dict, out_postings)
    indexer.index_collection(in_dir)

def compare(in_dir, out_dict, out_postings):
    print("Comparing in-memory hax and block impl")
    # This is an empty method
    # Pls implement your code in below
    # Try with hax
    indexer = Indexer(out_dict, out_postings)
    for _, _, files in os.walk(in_dir):
        for file in files:
            indexer.index(os.path.join(in_dir, file), int(file))
    indexer.finalize()
    A = indexer.get_full_postings()
    print(sys.getsizeof(A))
    MEMORY_LIMIT = int(1e6)

    # Try with SPIMI
    indexer = Indexer(out_dict, out_postings)
    indexer.index_collection(in_dir)
    B = indexer.get_full_postings()
    print(sys.getsizeof(B))

    # # Compare

    print(A == B)
    with open("debug.txt", "w") as outf:
        outf.write("Checking A\n")
        for k in A:
            if k not in B:
                outf.write(f"{k} is not in B but it is in A\n")
            elif A[k] != B[k]:
                outf.write(f"{len(A[k])} for {k} in A\n")
                outf.write(f"{A[k]} for {k} in A\n")
                outf.write(f"{len(B[k])} for {k} in B\n")
                outf.write(f"{B[k]} for {k} in A\n")
        for k in B:
            if k not in A:
                outf.write(f"{k} is not in B but it is in A\n")
            elif A[k] != B[k]:
                outf.write(f"{len(A[k])} for {k} in A\n")
                outf.write(f"{A[k]} for {k} in A\n")
                outf.write(f"{len(B[k])} for {k} in B\n")
                outf.write(f"{B[k]} for {k} in A\n")


def test_get_posting_lists(out_dict, out_postings):
    print("test get posting lists...")
    indexer = Indexer(out_dict, out_postings)
    indexer.load()
    for word in ["billion", "u.s.", "dollar", "week", ",", "mln"]:
        print(word)
        print(indexer.get_posting_list(word))
        print(indexer.word_to_pointer_dict[word])
        break


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
    test_get_posting_lists(output_file_dictionary, output_file_postings)
    # python3 index.py -i ./reuters/small-training -d dictionary.txt -p postings.txt
