This is the README file for A0235143N-A0233753E-A0236430N-A0235410W's submission
Email(s): e0735359@u.nus.edu, e0727410@u.nus.edu

== Python Version ==

I'm (We're) using Python Version 3.10.12 for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

The main class Dataset is responsible for loading the dataset as a stream of data elements, 
where each element represents a document with attributes like document ID, title, content, 
date posted, and court. We've implemented handling for duplicate rows by merging their court 
strings. Moreover, there's a method for tokenizing the content of each document, and we've 
included options to cache the tokenized data for improved performance. To preprocess the text 
data, we've utilized the preprocessor module, and for tracking progress, the code uses the 
tqdm library.

The index is constructed using a term frequency-inverse document frequency (TF-IDF) weighting scheme.
We load the dataset and tokenize the content of each document using the Dataset class.
For each document, we calculate TF-IDF weights for each term in the document and store 
these weights in a document vector, in which we construct an inverted index where each term
points to a list of (documentId, tf-idf weight) tuples. 

In search.py, we load the input query and relevant document IDs from the query file. Then,
we initialize the Indexer object with the dictionary and postings file paths. It parses the query
into tokens using QueryParser. We then create a Scorer object and initializes the query weights, 
applying pseudo-relevance feedback to adjust weights based on relevant and irrelevant documents.
Document scores are computed based on the adjusted query weights. We sort the documents 
by their scores, breaking ties by using document IDs. Finally, we write the sorted list to
the output file.


== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- README.txt: high level documentation
- index.py: index construction
- search.py: process queries and return ranked retrieval search result
- dictionary.txt: store dictionary mapping of token to file pointer
- postings.txt: store the posting lists of all tokens

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