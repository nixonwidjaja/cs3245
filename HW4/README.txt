This is the README file for A0235143N-A0233753E-A0236430N-A0235410W's submission
Email(s):
- e0727143@u.nus.edu
- e0725753@u.nus.edu
- e0735359@u.nus.edu
- e0727410@u.nus.edu

== Python Version ==

I'm (We're) using Python Version 3.10.12 for this assignment.

== General Notes about this assignment ==



# Overview

For our leaderboard submission, we used:
-   lnc.ltc ranking scheme (same as HW3), optimized by omitting dividing by
    query length (as it doesn't affect rankings).
-   WordNet query expansion, adding the synonyms of each word to the query
    tokens (only for free text queries).
-   Pseudo Relevance Feedback using top 5 scoring docs as relevant, and bottom
    100 docs as irrelevant, with alpha=0.9, beta=0.1, gamma=0.1.
-   Preprocessing:
    -   WordNet Lemmatization (with POS set to NOUN)
    -   Case-folding

We explored, but ultimately didn't use for the leaderboard:
-   Positional indices.
-   Stemming / Lemmatization with POS tagging.
-   Gap and Variable Byte encoding.
-   Weighing documents by their courts' importance.



# Execution flow for Indexing

1.  Use `csv` library to load the dataset row-by-row (done in `dataset.py`):
2.  For each document, we preprocess (done in `preprocessor.py`) by:
    1.  Using only the document's "content" text (ignoring other fields/zones).
    2.  Tokenize into sentences via `nltk.sent_tokenize`.
    3.  Tokenize each sentence via `nltk.word_tokenize`.
    4.  Lemmatize each token via `nltk.WordNetLemmatizer`
        (with POS tag set to default NOUN).
    5.  Case-fold to lowercase.
3.  For each document, we:
    1.  Compute its cosine-normalized log-TF weight vector.
    2.  Append the weights for each term in the postings list.
    3.  Save the document's weight vector (to be used in Pseudo RF).
4.  Once all documents have been processed, we save to the dictionary/postings
    files:
    1.  In dictionary file, store the pickled Python dict of:
        - inverted-index's term
        - term's DF
        - postings list offset/size (to retrieve from posting file)
        - document's offset/size (to retrieve from posting file)
    2.  In postings file, store the pickled Python lists of:
        - postings lists
        - document vectors



# Execution flow for Searching

1.  The terms, term's DF, offsets/sizes, and the document vectors' offset/sizes
    are loaded into memory via the `Indexer` class in `indexer.py`.
2.  We preprocess the query (done in `query_parser.py`) by:
    -   For boolean queries:
        1.  Split each AND seperated phrases and remove quotations (ie. ").
        2.  For each phrase, tokenize, lemmatize and case-fold like for the
            documents. No WordNet query expansion is done.
    -   For free text queries:
        1.  Tokenize query via `nltk.word_tokenize`.
        2.  Lemmatize each token via `nltk.WordNetLemmatizer`
            (with POS tag set to default NOUN).
        3.  Case-fold to lowercase.
        4.  For each token, add synonyms obtained from WordNet
            (with POS tag set to default NOUN).
3.  To retrieve a term's posting list (done in `indexer.py`):
    1.  Get the term's "offset" and "size" from the in-memory dict.
    2.  Seek to the "offset" byte in `postings.txt` file.
    3.  Read "size" number of bytes.
    4.  Load the bytes as a pickled Python list.
4.  The scores are computed (done in `scorer.py`) by:
    1.  Initializing query weights to it's log TF-IDF.
    2.  Pseudo Relevance Feedback is done with
        - alpha=0.9
        - beta=0.1
        - gamma=0.1
        And using the top 5 scoring documents as "relevant documents", and the
        bottom 100 scoring documents as "irrelevant documents".
5.  The scores are tie-broken by doc-ID and thedocument IDs are written to the
    output file.



# Other design decisions

1.  Caching of tokens.
    Since most of the indexing time is from tokenisation/text-preprocessing, we
    sped it up locally by saving computed tokens to a file (ie. a "cache") for
    subsequent indexing runs (for when we change our indexing method).

    Effects: Without the cache, it takes ~20mins to index. With it, it takes ~2mins.

2.  Streaming the data rows.
    To avoid loading the entire dataset/dataset's tokens, we operate on a stream
    (ie. a Python generator function) of data rows, loading each row into memory
    one-by-one instead of the entire dataset.

    Effects: Storing entire dataset and dataset's tokens in memory took ~10GB of
             RAM during indexing. Streaming each row only took ~1.3GB.

3.  Handling of duplicate Doc-IDs.
    There's some dataset rows with duplicate Doc-IDs which differ from each
    other by only their "court". This is handled by concatenating the "court"
    text of both, and removing the duplicates.

4.  No WordNet query expansion is done on boolean queries, as we assume the user
    knows the exact words he wants.



# Effects of Court Importance Weighing

We tried scaling the documents' log-TF based on the importance of their court
(done in `index.py`).

Results on the leaderboard are shown below, which showed that scaling worsened
our MAF2 score. So we didn't use court-importance scaling in the end.

## Without scaling
Leaderboard MAF2: 0.327


## Scaled by 1.0 / 0.5 / 0.25
Most important courts:  x1.0
Important courts:       x0.5
Others:                 x0.25
Leaderboard MAF2: 0.324


## Scaled by 1.0 / 0.8 / 0.6
Most important courts:  x1.0
Important courts:       x0.8
Others:                 x0.6
Leaderboard MAF2: 0.322



# Effects of WordNet query expansion

We observed that WordNet query expansion is essential for this corpus as, for example, sample q1 is "quiet phone call" but the relevant documents only have "silent telephone call" which do not match the query at all. Which lead to abyssmal
performance for q1:

Searching for the query "queries/q1.txt" ...
6807771   : Rank 3685
4001247   : Rank 154
3992148   : Rank 1642


With WordNet query expansion, the given relevant docs are ranked much higher:
Searching for the query "queries/q1.txt" ...
6807771   : Rank 401
4001247   : Rank 88
3992148   : Rank 392



# Effects of Pseudo Relevance Feedback

Pseudo relevance feedback improve our scores marginally, although the results
very sensitive on the hyperparameters (alpha, beta, gamma, number of docs used). 
In the end, we used the hyperparameters:

alpha = 0.9
beta = 0.1
gamma = 0.1
n_relevant = 5  (using top 5 scoring docs as relevant)
n_irrelevant = 100  (using bottom 100 docs as irrelevant)

Locally, the ranks of the given relevant docs with and without PRF shown below.
The ranks of some documents went up, and some went down, so it was difficult to
tune. It also took a lot longer to search, with q1 search time going up from
0.3s -> 5.2s.

However, it did increased our leaderboard MAF2 score from 0.319 -> 0.327.


Without Pseudo Relevance Feedback:
Searching for the query "queries/q1.txt" ...
Execution time: 0.3s
6807771   : Rank 401
4001247   : Rank 88
3992148   : Rank 392
Searching for the query "queries/q2.txt" ...
Execution time: 0.3s
2211154   : Rank 402
2748529   : Rank 13
Searching for the query "queries/q3.txt" ...
Execution time: 0.2s
4273155   : Rank 9
3243674   : Rank 2
2702938   : Rank 6


With Pseudo Relevance Feedback:
Searching for the query "queries/q1.txt" ...
Execution time: 5.2s
6807771   : Rank 354
4001247   : Rank 91
3992148   : Rank 448
Searching for the query "queries/q2.txt" ...
Execution time: 5.0s
2211154   : Rank 278
2748529   : Rank 15
Searching for the query "queries/q3.txt" ...
Execution time: 5.3s
4273155   : Rank 8
3243674   : Rank 2
2702938   : Rank 5



# Effects of preprocessing

We also experimented with 3 different preprocessing to see how well it worked
with WordNet query expansion:
- Stemming
- Lemmatization (with POS set to the default NOUN)
- Lemmatization (with POS tagging)

All the results below is using WordNet query expansion.
Lemmatization with NOUN POS worked best, while Lemmatization with inferred POS
tagging performed worst (we're not sure why tho).


## Stemming
Searching for the query "queries/q1.txt" ...
Execution time: 0.3s
6807771   : Rank 1029
4001247   : Rank 182
3992148   : Rank 1396
Searching for the query "queries/q2.txt" ...
Execution time: 0.3s
2211154   : Rank 411
2748529   : Rank 52
Searching for the query "queries/q3.txt" ...
Execution time: 0.2s
4273155   : Rank 12
3243674   : Rank 2
2702938   : Rank 5


## Lemmatization (with POS set to the default NOUN)
Searching for the query "queries/q1.txt" ...
Execution time: 0.3s
6807771   : Rank 401
4001247   : Rank 88
3992148   : Rank 392
Searching for the query "queries/q2.txt" ...
Execution time: 0.3s
2211154   : Rank 402
2748529   : Rank 13
Searching for the query "queries/q3.txt" ...
Execution time: 0.2s
4273155   : Rank 9
3243674   : Rank 2
2702938   : Rank 6


## Lemmatization (with POS tagging)
Searching for the query "queries/q1.txt" ...
Execution time: 0.4s
6807771   : Rank 3968
4001247   : Rank 106
3992148   : Rank 1258
Searching for the query "queries/q2.txt" ...
Execution time: 0.4s
2211154   : Rank 175
2748529   : Rank 141
Searching for the query "queries/q3.txt" ...
Execution time: 0.3s
4273155   : Rank 10
3243674   : Rank 2
2702938   : Rank 6



# Effects of Gap and Variable Byte Encoding

We also experimented with the usage of gap and variable byte encoding for the posting list, which can be enabled and disabled via a flag in the index.py and search.py scripts. When enabled, the size of the posting.txt reduces from 678MB to 575MB. However, the search time for our queries now also increase as we now need to decode the variable byte encoded doc ids and convert from the gap representation to the actual doc ids.

For example, the search time increased from:
query 1: 4.1s -> 5.5s
query 2: 3.6s -> 5.4s
query 3: 4.3s -> 5.2s

Because our current postings file without compression already satisfies the assignment's file size requirement, we chose not to use index compression techniques in our final submission as the time efficiency of our system would be more important.

We started with a simple model that completely ignored the positional queries and obtained a mean average F2 score of 0.327139583308408 on the leaderboards. Afterwards, we decided to work on implementing the positional indexing.

The first issue we ran into was the size of the positional postings list, which was ~900MB, greatly over the maximum size of the submission folder. We attempted to use the posting list compression techniques but the indexing stalled due to a lack to memory and took forever. Furthermore, our posting list compression technique has less savings for positional indices. This is because we can choose to store our positional postings as a dictionary of terms to another dictionary of positional indexes to the posting list of documents. The same document can now occur multiple times for the same term, furthermore each sub posting list for each positional index is smaller than without, thus we incur less savings using our variable byte encoding. The gap encoding is also not efficient when the gaps are less than 2**30-1 due to how numbers are represented in python. The alternative representation for the positional indexing would be to store the postings as a map from term to a list of tuples which contain doc id and positional id. This is strictly inferior in every aspect: it makes the positional search itself extremely inefficient because we are unable to index their positional differences using the map and are forced to iterate through all possible combinations. Storage is also significantly more expensive as the document id needs to be repeated for each occurrence. As such, we decided to stick with our current implementation and ignore the positional indexing since our submission was sufficiently good (on the leaderboards at least).



== Allocation work ==

- Shaun:    Wrote main bulk of the code, submitted our code to leaderboard,
            experimented with Pseudo RF, query expansion, weighing court
            importance, updated README.
- Neale:    Explored possible ideas, tested our code on SoC slurm.
- Jotham:   Wrote code for all the compression techniques, explored using
            positional index, wrote the bonus.docx, updated README.
- Nixon:    Drafted the README documentation.



== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- README.txt:       High level documentation.
- index.py:         Script for indexing.
- search.py:        Script for searching queries.
- dictionary.txt    Contains pickled Python dicts of:
                        1.  Terms, terms' DF, byte-offsets/sizes for the each
                            terms' corresponding posting list in 'postings.txt'.
                        2.  Byte-offsets/sizes for each documents' vector in
                            'postings.txt'.
- postings.txt      Contains pickled Python lists of:
                        1.  Posting lists for each term, formatted as
                            `[(doc_id, doc_normalized_weight), ...]`.
                        2.  Weight vector for each document, formatted as
                            `[{ term: weight }, ...]`
- preprocessor.py:  Handles preprocessing of documents' text.
- query_parser.py:  Handles parsing of queries, and WordNet query expansion.
- scorer.py:        Handles computing query weights, documents' ranking scores,
                    and performing Pseudo Relevance-Feedback.
- indexer.py:       Handles reading the dictionary and postings files.
- dataset.py:       Handles the loading of dataset.
- bonus.docx:       Writeup for the bonus component.



== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] We, A0235143N-A0233753E-A0236430N-A0235410W, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

We suggest that we should be graded as follows:

== References ==

The Online IR textbook
Lecture notes: algorithms needed