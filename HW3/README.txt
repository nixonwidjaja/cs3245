This is the README file for A0236430N-A0235410W's submission
Email(s): e0735359@u.nus.edu, e0727410@u.nus.edu

== Python Version ==

I'm (We're) using Python Version 3.10.12 for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

Our ranked retrieval system comprises of two main components, the indexing and the searching.

Our indexing takes an input directory that comprises of documents identified by 
an integer. It uses NLTK's sent_tokenize, word_tokenize, PorterStemmer 
and lower case folding to tokenize the documents. It then computes a Posting abstraction for 
each document which includes the term frequency of a term for each document.
Afterwards, the log(tf) + 1 values of the Posting are pre-computed to speed up 
the search algorithm and stored in a posting files. The document vector for each document is 
also pre-computed and stored in a separate document lengths file.
Finally, the dictionary itself is pickled into a dictionary binary file.

In searching, we first apply the exact same preprocessing step as used by indexing on our 
query and then convert the query string into a Counter of term frequencies in the query, 
which we use to compute the query vector where tf is computed using the term frequency with 
respect to the Counter and idf is computed with respect to the indexed corpus. We follow the 
optimised cosine score algorithm as shown in the lecture closely.
We retrieve the Posting List for each unique query term. The score for each document is 
computed with the precomputed w_t_d which was log base 10 of term freq + 1 and the query 
weight in the query vector. Finally, the document score is normalized using the precomputed 
document length in the indexing stage. The heap data structure is then used to optimise the 
top-K retrieval. As far as possible, we have optimized our search to use as much 
pre-computation as possible and ended up with a total run time of ~38 seconds on the sanity queries.


== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- README.txt: high level documentation
- index.py: index construction
- search.py: process queries and return ranked retrieval search result
- dictionary.txt: store dictionary mapping of token to file pointer
- postings.txt: store the posting lists of all tokens
- lengths.txt: stores the precomputed document vector length

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] I/We, A0236430N-A0235410W, certify that I/we have followed the CS 3245 Information
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

Piazza - Jed, Shaun Tan, Zhao Jin
The Online IR textbook
Lecture notes: algorithms needed