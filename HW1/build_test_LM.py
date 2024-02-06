#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import sys
import getopt

LANGUAGES = ['indonesian', 'malaysian', 'tamil']

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print("building language models...")
    # This is an empty method
    # Pls implement your code below

    # Open input file and data will be a list containing each line in the file as a list item
    with open(in_file, 'r') as f:
        data = f.readlines()
    # Initialize LM (language model) with final format: {language: {(gram tuple): probability}}
    lm = {}
    for lan in LANGUAGES:
        lm[lan] = {}
    for line in data:
        # Remove \n at the end of line
        line = line[:-1]
        # Split the first word (language name) and the rest of the sentence
        language, sentence = line.split(' ', 1)
        # Convert each character in the sentence into a separate element in a list
        chars = list(sentence)
        for i in range(len(chars) - 3):
            # Define gram as a tuple of 4 characters obtained by sliding window
            gram = tuple(chars[i:(i+4)])
            # Add one smoothing: set the count of the gram = 1 for all languages
            for lan in lm:
                if gram not in lm[lan]:
                    lm[lan][gram] = 1
            # Add the count for the gram to its respective language
            lm[language][gram] += 1
    # Convert count into probability
    for lan in lm:
        # total_count represents the total count of all grams in a language
        total_count = 0
        for g in lm[lan]:
            total_count += lm[lan][g]
        # Divide each gram count with the language's total count
        for g in lm[lan]:
            lm[lan][g] /= total_count
    return lm

def evaluate(sentence, LM):
    """
    Helper function that returns the language prediction of a sentence based on a given LM.
    """
    # Remove \n at end of line and convert each character in the sentence into a separate element in a list
    s = list(sentence[:-1])
    miss = 0
    count = 0
    # Initialize probability dict with final format: {language: probability}
    probability = {}
    for lan in LANGUAGES:
        probability[lan] = 0
    for i in range(len(s) - 3):
        # Define gram as a tuple of 4 characters obtained by sliding window
        gram = tuple(s[i:(i+4)])
        for lan in LM:
            # Count the total gram count of each sentence
            count += 1
            # If the gram exists in given LM, add the log probability to its respective language in probability dict
            if gram in LM[lan]:
                probability[lan] += math.log(LM[lan][gram])
            # Else, ignore the gram but increment miss count as an indicator of alien language
            else:
                miss += 1
    # If there are more grams unrecognized in LM than otherwise, label as alien language
    if miss / count > 0.5:
        return 'other'
    # Return the language that has the max probability out of all language options
    language = max(list(probability.items()), key=lambda x: x[1])[0]
    return language
        

def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")
    # This is an empty method
    # Pls implement your code below
    with open(in_file, 'r') as f:
        data = f.readlines()
    # Initialize the answer list
    ans = []
    # For each sentence in test file, find the most probable language and append to answer list
    for sentence in data:
        pred = evaluate(sentence, LM) 
        ans.append(pred + ' ' + sentence)
    # Write the answer into the output file
    with open(out_file, 'w') as f_out:
        f_out.writelines(ans)


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file"
    )


input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], "b:t:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == "-b":
        input_file_b = a
    elif o == "-t":
        input_file_t = a
    elif o == "-o":
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
