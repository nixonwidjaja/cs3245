"""Test case for testing whether all doc ids are stored in the empty string"""
import os

from index import Indexer, UNIVERSE
# These imports are necessary for Pickle.load
# Python needs to know what classes are being deserialized into so we need
# to load the classes into memory for Pickle to work
from index import WordToPointerEntry, PostingsList, Posting


d = "dictionary.txt"
p = "postings.txt"
indexer = Indexer(out_dict=d, out_postings=p)

expected = []
for _, _, files in os.walk("./reuters/training/"):
    for file in files:
        expected.append(int(file))
expected = sorted(expected)
actual = [posting.value for posting in indexer.get_posting_list("")]

assert actual == expected