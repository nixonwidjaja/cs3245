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
    with open(in_file, 'r') as f:
        data = f.readlines()
    lm = {}
    for lan in LANGUAGES:
        lm[lan] = {}
    for s in data:
        s = s[:-1]
        language, sentence = s.split(' ', 1)
        d = list(sentence)
        for i in range(len(d) - 3):
            gram = tuple(d[i:(i+4)])
            for lan in lm:
                lm[lan][gram] = 1
    for s in data:
        s = s[:-1]
        language, sentence = s.split(' ', 1)
        d = list(sentence)
        for i in range(len(d) - 3):
            gram = tuple(d[i:(i+4)])
            lm[language][gram] += 1
    for lan in lm:
        count = 0
        for g in lm[lan]:
            count += lm[lan][g]
        for g in lm[lan]:
            lm[lan][g] /= count
    return lm

def evaluate(sentence, LM):
    s = list(sentence[:-1])
    miss = 0
    count = 0
    probability = {}
    for lan in LANGUAGES:
        probability[lan] = 0
    for i in range(len(s) - 3):
        gram = tuple(s[i:(i+4)])
        for lan in LM:
            count += 1
            if gram in LM[lan]:
                probability[lan] += math.log(LM[lan][gram])
            else:
                miss += 1
    if miss / count > 0.5:
        return 'other'
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
    ans = []
    for sentence in data:
        pred = evaluate(sentence, LM) 
        ans.append(pred + ' ' + sentence)
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
