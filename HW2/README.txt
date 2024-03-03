This is the README file for A0236430N-A0235410W's submission
Email(s): e0735359@u.nus.edu, e0727410@u.nus.edu

== Python Version ==

I'm (We're) using Python Version 3.10.12 for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

Our Boolean Retrieval indexing system comprises of two main components, the indexing and the searching.

Our indexing makes use of the Single-pass in-memory indexing (SPIMI) technique. 
Given an input directory that comprises of documents identified by an integer, 
we use NLTK's sent_tokenize, word_tokenize, and PorterStemmer stemming to tokenize the documents 
into a token stream. We then lazily iterate over each token and build up our Index, 
writing the dictionary and posting list to a new block file when our memory limit is reached. 
After successfully writing all tokens to the hard disk, we apply n-way merging on all the 
blocks to generate our final dictionary and postings file, making sure to add skip pointers 
to each posting list for faster search as well as posting list size for subsequent optimisation. 
We also create an additional posting list to represent the textual corpus to handle NOT queries easier.

In our search algorithm, for each query, we split the sentence into words and use PorterStemmer
stemming to tokenize the query. We then use the Shunting-Yard algorithm to parse 
the textual query into a valid boolean postfix syntax and apply the skip pointer to process
AND, OR, and NOT operators. We implement 2 algorithms for query evaluation: naive_search() and 
search(). We made sure both algorithms are correct and produce the same results. naive_search() naively
process each token ordered by Shunting Yard without any optimisation, while search() groups together 
ANDs of the same "level" and sort them in the order of increasing posting list length. We timed each
algorithm and interestingly, for given sanity-queries.txt, 
search() takes twice as long as naive_search() to complete. One possible
reason is that search() reprocesses a whole query using a stack one more time on top of Shunting Yard to 
group the ANDs and sort them, whereas naive_search() directly evaluate the posting lists. Another 
reason is that sanity-queries.txt has too short queries such that the extra preprocessing done
by search() cannot show significant performance improvement. Longer queries may benefit from this
optimisation.

We approached our development incrementally, first starting with a completely in-memory 
indexing technique as our source of truth to validate our SPIMI implementation. We then 
validated our search algorithm by manually tracing through on a small subset of Reuters to 
validate the correctness of our search algorithm. We further discussed with our fellow peers on Piazza.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- README.txt: high level documentation
- index.py: index construction
- search.py: process queries and return search result
- dictionary.txt: store dictionary mapping of token to file pointer
- postings.txt: store the posting lists of all tokens

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] We, A0236430N-A0235410W, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

50/50 correctness of code
20/20 documentation
30/30 query evaluation

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

Piazza - Jed, Malcom, Kenneth, Zhao Jin: clarifications on preprocessing and data structure
The Online IR textbook
Lecture notes: algorithms needed