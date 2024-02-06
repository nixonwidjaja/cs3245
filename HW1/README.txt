This is the README file for A0236430N's submission
Email: e0735359@u.nus.edu

== Python Version ==

I'm using Python Version 3.9.7 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

High level overview:
- To build the LM, I used a dictionary of dicts with format: {language: {(gram tuple): probability}}
because it's fast to get and store data (O(1)) and can accept string, tuples, and number
- For each train sentences, I split the sentence into 4-grams. Since we're using add one smoothing,
I need to add the gram to all languages regardless if it exists in the language or not, then set the 
count as 1. I added the counts of the gram to its respective language.
- After all the gram counts are set, for each language, find the total count of grams in each language 
and divide every gram counts with its total count to get the probability of each grams.
- LM containing the probability of each gram in each language is ready to be used.
- To predict the language of a test sentence, split the sentence into 4-grams and calculate the 
probability of that sentence. The language with the max probability will be the answer.
- However, the bigger the LM, the smaller the probability of each gram would be. Using math.log()
would be wise in this situation.
- To decide if a sentence is from alien language, I count the number of unrecognized grams in 
that sentence with respect to given LM. If there are more unrecognized grams than otherwise, 
it will label the sentence as 'other'.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

- build_test_LM.py  : source code with detailed documentation
- README.txt        : high level overview of code and statement of individual work
- ESSAY.txt         : answer to essay questions

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I, A0236430N, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0236430N, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:
- (50/50) Correctness
- (20/20) Documentation

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

- Lecture notes
