#!/usr/bin/python3
from index import Indexer, Posting, PostingsList

import re
import nltk
import sys
import getopt

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

"""
BOOLEAN EXPRESSION PARSING
"""
# FIXME: Temporary ? There is probably a better way to structure the query tree

class AND:
    def __init__(self, queries):
        self.queries = queries
        
    def add(self, query):
        self.queries.append(query)
        return self
        
    def __repr__(self):
        return "AND [ " + str(self.queries) + " ] "
    
    
class OR:
    def __init__(self, queries):
        self.queries = queries
        
    def add(self, query):
        self.queries.append(query)
        return self

    def __repr__(self):
        return "OR [ " + str(self.queries) + " ] "


class AND_NOT:
    def __init__(self, AND, NOT):
        self.AND = AND
        self.NOT = NOT
    
    def __repr__(self):
        return f"[{self.AND}] AND NOT [{self.NOT}]"
    

class NOT:
    def __init__(self, query):
        self.query = query
    
    def __repr__(self):
        return f"NOT [ {self.query} ] "        


def split(q):
    """
    Split the query into their individual tokens.
    We are adding whitespaces between the parentheses to make it easy to recursively
    shunting yard on the expression later so that the parentheses themselves count
    as a single token.
    We are also splitting by keywords AND, OR, NOT to separate the tokens.
    For example, queries such as 'bunny balls' should be counted as a single token.
    """
    q = re.sub(r'[(]', "( ", q)
    q = re.sub(r'[)]', " )", q)
    tokens = q.split()
    new_tokens = []
    curr_token = ""
    for token in tokens:
        if token.upper() in ["AND", "OR", "NOT", "(", ")"]:
            # if the query is syntatically correct, curr_token will not be ""
            # Possible for curr_token to be "" if its AND NOT
            if curr_token != "":
                new_tokens.append(curr_token)
                curr_token = ""
            new_tokens.append(token)
        elif curr_token == "":
            curr_token = token
        else:
            curr_token = curr_token + " " + token
    if curr_token != "":
        new_tokens.append(curr_token)
    print(str(new_tokens))
    return new_tokens
    
    
def shunting(tokens):
    """Given a sequence of tokens, return a postfix syntax representation according to
    Shunting Yard algorithm.
    """
    if not tokens:
        return
    operator_stack = []
    result_stack = []
    
    def flush():
        nonlocal operator_stack, result_stack
        while operator_stack:
            result_stack.append(operator_stack.pop())
    
    PRECEDENCE = {
        "NOT": 4,
        "AND NOT": 3,
        "AND": 2,
        "OR": 1,
    }
    
    operators = ["OR", "AND", "NOT"]
    # Apply the Shunting Yard algorithm
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.upper() in operators:
            token = token.upper()
            if token == "AND" and tokens[i+1].upper() == "NOT":
                token = "AND NOT"
                i += 1
            if operator_stack and PRECEDENCE[operator_stack[-1]] > PRECEDENCE[token]:
                flush()
            operator_stack.append(token)
        elif token == "(":
            right_parenth_idx = tokens[i:].index(")") + i
            result = shunting(tokens[i+1:right_parenth_idx])
            i = right_parenth_idx
            for t in result:
                result_stack.append(t)
        else:
            result_stack.append(token)
        i += 1
    flush()
    return result_stack
    

def optimize_ast(query):
    """
    Right now, this is very makeshift and mostly uses my own algorithm which may or
    may not work well, or even correctly but it's a trial and error thing at the moment.
    We assume that the query is already in postfix notation.
    Assumes that it is in the postfix notation.
    Clearly, we want to combine all terms in the AND operand.
    """
    print("Optimizing " + str(query))
    operand_stack = []
    for token in query:
        print(operand_stack)
        if token in ["AND", "OR", "AND NOT"]:
            s1 = operand_stack.pop()
            s2 = operand_stack.pop()
            print(s1)
            print(s2)
            if token == "AND":
                if isinstance(s1, AND):
                    operand_stack.append(s1.add(s2))
                elif isinstance(s2, AND):
                    operand_stack.append(s2.add(s1))
                else:
                    operand_stack.append(AND([s1, s2]))
            elif token == "OR":
                if isinstance(s1, OR):
                    operand_stack.append(s1.add(s2))
                else:
                    operand_stack.append(OR([s1, s2]))
            else:  #token == "AND NOT":
                operand_stack.append(AND_NOT(AND=s2, NOT=s1))
        elif token in ["NOT"]:
            s1 = operand_stack.pop()
            operand_stack.append(NOT(s1))
        else:
            operand_stack.append(token)
    assert len(operand_stack) == 1
    return operand_stack[0]


def parse_query(query):
    """Given a boolean query, convert it into an optimized ast"""
    # return shunting(split(query))
    return optimize_ast(shunting(split(query)))

"""
BOOLEAN OPERATORS
"""

def apply_and(indexer: Indexer, term1: str, term2: str) -> list[str]:
    """
    Apply the AND operation on term 1 and term 2 using low level operations.
    We assume that the posting list file for both term exists.
    Make use of skip pointers whenever possible
    """
    pl1: PostingsList = indexer.get_posting_list(term1)
    pl2: PostingsList = indexer.get_posting_list(term2)
    p1 = p2 = 0  # Pointers to each posting list
    results = []
    while p1 < len(pl1) and p2 < len(pl2):
        if pl1[p1] == pl2[p2]:
            results.append(pl1[p1].value)
            p1 += 1
            p2 += 1
        elif pl1[p1] < pl2[p2]:
            while pl1[p1].has_skip() and pl1[pl1[p1].skip] <= pl2[p2]:
                p1 = pl1[p1].skip
            else:
                p1 += 1
        else:
            # We gotta do this nested if and while because we want to be able
            # to match on the equality with the skip pointer and not do the
            # increment. Consider the following case
            # 1 -> 5 -> 10 -> 15 -> 20 (with a skip from 1 to 10 to 20)
            # 20
            # we should skip from 1 to 10 and to 20 and then not do the increment
            # doing a while else, while it looks neat, will miss out the 20
            # This mirrors the algorithm shown in the lecture notes
            if pl2[p2].has_skip() and pl2[pl2[p2].skip] <= pl1[p1]:
                while pl2[p2].has_skip() and pl2[pl2[p2].skip] <= pl1[p1]:
                    p2 = pl2[p2].skip
            else:
                p2 += 1
    return results


def apply_or(indexer: Indexer, term1: str, term2: str) -> list[str]:
    """
    Apply the OR operation on term 1 and term 2 using low level operations.
    We assume that the posting list file for both term exists.
    Skip pointers are useless for or
    """
    pl1: PostingsList = indexer.get_posting_list(term1)
    pl2: PostingsList = indexer.get_posting_list(term2)
    p1 = p2 = 0  # Pointers to each posting list
    results = []
    while p1 < len(pl1) and p2 < len(pl2):
        if pl1[p1] == pl2[p2]:
            results.append(pl1[p1].value)
            p1 += 1
            p2 += 1
        elif pl1[1] < pl2[p2]:
            results.append(pl1[p1].value)
            p1 += 1
        else:
            results.append(pl2[p2].value)
            p2 += 1
    while p1 < len(pl1):
        results.append(pl1[p1].value)
        p1 += 1
    while p2 < len(pl2):
        results.append(pl2[p2].value)
        p2 += 1
    return results


def apply_and_not(indexer: Indexer, term1: str, term2: str) -> list[str]:
    """
    Apply the term1 AND NOT term2
    term1: 1 2 3 4 5
    term2: 2 3 
    """
    pl1: PostingsList = indexer.get_posting_list(term1)
    pl2: PostingsList = indexer.get_posting_list(term2)
    p1 = p2 = 0  # Pointers to each posting list
    results = []
    while p1 < len(pl1) and p2 < len(pl2):
        if pl1[p1] < pl2[p2]:
            results.append(pl1[p1].value)
            p1 += 1
        elif pl1[p1] == pl2[p2]:
            p1 += 1
            p2 += 1
        else:
            # Same reasoning as AND skip pointer
            # We still want to jump to equality so that we can advance p1
            if pl2[p2].has_skip() and pl2[pl2[p2].skip] <= pl1[p1]:
                while pl2[p2].has_skip() and pl2[pl2[p2].skip] <= pl1[p1]:
                    p2 = pl2[p2].skip
            else:
                p2 += 1
    while p1 < len(pl1):
        results.append(pl1[p1].value)
        p1 += 1
    return results


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    q = "bill OR Gates AND (vista OR XP) AND NOT mac"
    print(parse_query(q))
    # # We cannot read the whole posting files into memory
    # indexer = Indexer(dict_file, postings_file)
    # with open(results_file, "w") as outf, open(queries_file, "r") as inf:
    #     query_no = 0
    #     # Process each query and write to file
    #     while True:
    #         query = inf.readline()
    #         query = query.strip()
    #         if not query:
    #             break
    #         print("Original query is " + query)
    #         query = parse_query(query)
    #         print("New query is " + str(query))

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
