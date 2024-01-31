#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import sys
import getopt


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
    lm['indonesian'] = {}
    lm['malaysian'] = {}
    lm['tamil'] = {}
    for s in data:
        s = s[:-1]
        sentence = s.split(' ')
        language = sentence[0]
        d = list(' '.join(sentence[1:]))
        for i in range(len(d) - 3):
            gram = tuple(d[i:(i+4)])
            for lan in lm:
                lm[lan][gram] = 1
    for s in data:
        s = s[:-1]
        sentence = s.split(' ')
        language = sentence[0]
        d = list(' '.join(sentence[1:]))
        for i in range(len(d) - 3):
            gram = tuple(d[i:(i+4)])
            lm[language][gram] += 1
    count = {}
    for lan in lm:
        count[lan] = 0
        for g in lm[lan]:
            count[lan] += lm[lan][g]
    for lan in lm:
        for g in lm[lan]:
            lm[lan][g] = lm[lan][g] / count[lan]
    # for lan in lm:
    #     print(len(lm[lan]))
    #     sum = 0
    #     for g in lm[lan]:
    #         sum += lm[lan][g]
    #     print(sum)
    return lm

def evaluate(sentence, LM):
    s = list(sentence[:-1])
    prediction = {}
    prediction['indonesian'] = []
    prediction['malaysian'] = []
    prediction['tamil'] = []
    for i in range(len(s) - 3):
        gram = tuple(s[i:(i+4)])
        for lan in LM:
            if gram in LM[lan]:
                prediction[lan].append(LM[lan][gram])
    if len(prediction['indonesian']) == 0:
        return 'other'
    total = {}
    total['indonesian'] = 1
    total['malaysian'] = 1
    total['tamil'] = 1
    for lan in prediction:
        for i in prediction[lan]:
            total[lan] *= i
    return max(list(total.items()), key=lambda x: x[1])[0]
        

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
