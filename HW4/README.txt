This is the README file for A0235143N-A0233753E-A0236430N-A0235410W's submission
Email(s): e0735359@u.nus.edu, e0727410@u.nus.edu

== Python Version ==

I'm (We're) using Python Version 3.10.12 for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

Our program is an information retrieval system designed for legal case retrieval and comprises of two components: an indexer and a searcher.

Given a path to the dataset csv file, we first stream each row of the corpus using dataset.py. We use python's csv.reader file to iterate through each element in the legal corpus, retrieving its document_id, title, content, date_posted and court. Some rows may be duplicated and next to one another (they share the same document id but differ in their court), in this case the court is appended and the row is yielded lazily.

Afterwards, the preprocessor.py tokenizes each legal document's content with 3 preprocessing modes: "stem", "lemma_wo_pos" and "lemma_with_pos". All 3 preprocessing modes employ nltk's sent_tokenize, word_tokenize and differ afterwards. "stem" makes use of PorterStemmer::stem while "lemma_wo_pos" employs nltk.WordNetLemmatizer without part of speech (POS tag) set to noun. "lemma_with_pos" further uses nltk's recommended POS tagger to tag the list of tokens, converts them to WordNet's POS tag before using the nltk.WordNetLemmatizer. Finally, case folding is applied for all preprocessing methods.

We then proceed to construct our inverted index given each document id and its list of tokens as well as the document vectors which contains the pre-computed normalised vector using the L2 norm. We use pickle to store our index in a postings file. We also store the "metadata" for each term and document in our dictionary file to facilitate retrieval of term and doc. The metadata includes the df (length of posting list in inverted index) for terms, starting offset and size for file pointer for both terms and docs.

At querying/searching time, we accept the query and the list of relevant doc ids. We load the entire dictionary and postings file into memory and make use of our QueryParser class to obtain the query tokens.

Our QueryParser class handles the preprocessing of tokens, including the query-expansion technique. The preprocessing defers to our Preprocessor class in preprocessor.py to ensure that the query preprocessing matches with the index preprocessing. Query Expansion is performed using Wordnet's Synset which is an interface representing groups of synonymous words that express the same concept and the WordNet's POS tagged token.

After the list of query tokens (with query expansion) is obtained, they are passed to our Scorer class which calculates the scoring using the tf-idf scoring scheme alongside relevance feedback. The term weights are first initialized, by loading the precomputed normalised df score for each term, while dropping terms that don't appear in any docs. The query weights are computed using logarithm for the tf and idf for the normalised df.

Next, pseudo relevance feedback is optionally performed where the top n documents are assumed to be relevant and the rocchio formula is used to update the query weights.

After the query weights have been confirmed, the document scores are computed using the optimised cosine-scoring algorithm as taught in the lecture. The scores are tie-broken by doc-ID and written to the output file.

After testing on the sample queries, we opted for query expansion as our only form of query refinement (even though we had implemented pseudo relevance feedback as well). Firstly, we observed that query expansion is essential for this corpus as sample query 1 is  "quiet phone call" but the relevant documents only have "silent telephone" which do not match the query at all. Furthermore, although pseudo relevance feedback did improve our scores marginally, the results were too dependent on the hyperparameters (alpha, beta, gamma, number of docs) which we were unable to tune accordingly due to the lack of a sufficiently large training and validation set.

We also experimented with our three different preprocessing modes on the sample queries.

- Stemming preprocessing (did not use query expansion as stemming is not compatible)

Searching for the query "queries/q1.txt" ...
Execution time: 0.2s
6807771   : Rank 4469
4001247   : Rank 248
3992148   : Rank 3791
Searching for the query "queries/q2.txt" ...
Execution time: 0.2s
2211154   : Rank 90
2748529   : Rank 1
Searching for the query "queries/q3.txt" ...
Execution time: 0.3s
4273155   : Rank 18
3243674   : Rank 2
2702938   : Rank 16

- Lemmatization + Query Expansion

Searching for the query "queries/q1.txt" ...
Execution time: 1.6s
6807771   : Rank 2785
4001247   : Rank 62
3992148   : Rank 543
Searching for the query "queries/q2.txt" ...
Execution time: 1.6s
2211154   : Rank 75
2748529   : Rank 24
Searching for the query "queries/q3.txt" ...
Execution time: 1.7s
4273155   : Rank 5
3243674   : Rank 3
2702938   : Rank 9

- Lemmatization + POS tagging + Query Expansion

Searching for the query "queries/q1.txt" ...
Execution time: 1.6s
6807771   : Rank 4200
4001247   : Rank 88
3992148   : Rank 1373
Searching for the query "queries/q2.txt" ...
Execution time: 1.7s
2211154   : Rank 108
2748529   : Rank 32
Searching for the query "queries/q3.txt" ...
Execution time: 1.6s
4273155   : Rank 7
3243674   : Rank 3
2702938   : Rank 8

We also experimented with the usage of gap and variable byte encoding for the posting list, which can be enabled and disabled via a flag in the index.py and search.py scripts. When enabled, the size of the posting.txt reduces from 678MB to 575MB. However, the search time for our queries now also increase as we now need to decode the variable byte encoded doc ids and convert from the gap representation to the actual doc ids. For example, query 1 goes from 4.1s to 5.5s, query 2 goes from 3.6s to 5.4s and query 3 goes from 4.3s to 5.2s. Because our current postings file without compression already satisfies the assignment requirements, we chose not to use index compression techniques in our final submission as the time efficiency of our system would be more important.

We started with a simple model that completely ignored the positional queries and obtained a mean average F2 score of 0.327139583308408 on the leaderboards. Afterwards, we decided to work on implementing the positional indexing.

The first issue we ran into was the size of the positional postings list, which was ~900MB, greatly over the maximum size of the submission folder. We attempted to use the posting list compression techniques but the indexing stalled due to a lack to memory and took forever. Furthermore, our posting list compression technique has less savings for positional indices. This is because we can choose to store our positional postings as a dictionary of terms to another dictionary of positional indexes to the posting list of documents. The same document can now occur multiple times for the same term, furthermore each sub posting list for each positional index is smaller than without, thus we incur less savings using our variable byte encoding. The gap encoding is also not efficient when the gaps are less than 2**30-1 due to how numbers are represented in python. As such, we decided to stick with our current implementation and ignore the positional indexing since our submission was sufficiently good (on the leaderboards at least).

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- README.txt: high level documentation
- index.py: index construction
- search.py: process queries and return ranked retrieval search result
- dictionary.txt: store dictionary mapping of token to file pointer
- postings.txt: store the posting lists of all tokens
- preprocessor.py: Text preprocessing class
- query_parser.py: parses the queries
- scorer.py: scores the candidate documents according to consine distance
- indexer.py: class that facilitates access to the dictionary and postings list
- dataset.py: lazily loads and generates the token stream
- bonus.docx: our submission for the bonus component

== Allocation work ==
- Shaun: 
- Neale:
- Jotham:
- Nixon: README documentation

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] We, A0235143N-A0233753E-A0236430N-A0235410W, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

The Online IR textbook
Lecture notes: algorithms needed