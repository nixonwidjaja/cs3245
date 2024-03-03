#!/usr/bin/python3
from index import Indexer, UNIVERSE

# These imports are necessary for Pickle.load
# Python needs to know what classes are being deserialized into so we need
# to load the classes into memory for Pickle to work
from index import WordToPointerEntry, PostingsList, Posting

import math
import nltk
import re
import sys
import getopt
import time


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"
    )


"""
BOOLEAN EXPRESSION PARSING
"""


def split(q, stemmer: nltk.stem.PorterStemmer):
    """
    Split the query into their individual tokens.
    We are adding whitespaces between the parentheses to make it easy to recursively
    shunting yard on the expression later so that the parentheses themselves count
    as a single token.
    We are also splitting by keywords AND, OR, NOT to separate the tokens.
    For example, queries such as 'bunny balls' should be counted as a single token.
    Returns None if invalid
    """
    q = re.sub(r"[(]", "( ", q)
    q = re.sub(r"[)]", " )", q)
    tokens = q.split()
    new_tokens = []
    regular_term_count = 0
    if tokens[0] in ["AND", "OR"] or tokens[-1] in ["AND", "OR", "NOT"]:
        return None
    for token in tokens:
        if token.upper() in ["AND", "OR", "NOT", "(", ")"]:
            regular_term_count = 0
            new_tokens.append(token.upper())
        else:
            regular_term_count += 1
            if regular_term_count > 1:
                return None
            new_tokens.append(stemmer.stem(token.lower(), to_lowercase=True))
    return new_tokens


def shunting(tokens) -> list[str]:
    """Given a sequence of tokens, return a postfix syntax representation according to
    Shunting Yard algorithm.
    """
    if not tokens:
        return
    operator_stack = []
    result_stack = []

    PRECEDENCE = {
        "NOT": 3,
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
            while operator_stack and PRECEDENCE[operator_stack[-1]] > PRECEDENCE[token]:
                result_stack.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == "(":
            right_parenth_idx = tokens[i:].index(")") + i
            result = shunting(tokens[i + 1 : right_parenth_idx])
            i = right_parenth_idx
            for t in result:
                result_stack.append(t)
        else:
            result_stack.append(token)
        i += 1
    while operator_stack:
        result_stack.append(operator_stack.pop())
    return result_stack


def parse_query(query, preprocessing_fn: callable):
    """Given a boolean query, convert it into an optimized ast using shunting and return nothing if it's
    an illegal query term i.e. words with spaces."""
    query = preprocessing_fn(query)
    if isinstance(query, list):
        query = " ".join(query)
    print("Query after preprocessing is: " + query)
    return shunting(split(query))


"""
BOOLEAN OPERATORS
"""


def convert_posting_to_list(result: list[int]) -> PostingsList:
    """need to recreate the Posting"""
    pl = PostingsList()
    pl.plist = [Posting(docId) for docId in result]
    return pl


def reapply_skip_pointers(pl: PostingsList) -> PostingsList:
    """
    When merging two posting lists, their old skip pointers will
    no longer be valid as it is possible we take a posting from one
    posting list that points to an invalid index!
    We will apply a linear time reapplication of the skip pointers.
    There is no need to sort because we did our intersection in ascending order
    anyways.
    We will not be reusing PostingLists add skip pointers because we don't need to sort again
    """
    for p in pl:
        p.skip = None
    skips = round(math.sqrt(len(pl)))
    if skips == 0:
        return pl
    step = len(pl) // skips
    for i in range(0, len(pl), step):
        if i + step < len(pl):
            pl[i].skip = i + step
    return pl


def apply_and(pl1: PostingsList, pl2: PostingsList) -> PostingsList:
    """
    Apply the AND operation on term 1 and term 2 using low level operations.
    We assume that the posting list file for both term exists.
    Make use of skip pointers whenever possible
    """
    p1 = pl1.initialise_linked_list()
    p2 = pl2.initialise_linked_list()
    results = []
    while p1 is not None and p2 is not None:
        if p1 == p2:
            results.append(p1.value)
            p1 = p1.next
            p2 = p2.next
        elif p1 < p2:
            if p1.has_skip() and p1.skip_ptr <= p2:
                while p1.has_skip() and p1.skip_ptr <= p2:
                    p1 = p1.skip_ptr
            else:
                p1 = p1.next
        else:
            # We gotta do this nested if and while because we want to be able
            # to match on the equality with the skip pointer and not do the
            # increment. Consider the following case
            # 1 -> 5 -> 10 -> 15 -> 20 (with a skip from 1 to 10 to 20)
            # 20
            # we should skip from 1 to 10 and to 20 and then not do the increment
            # doing a while else, while it looks neat, will miss out the 20
            # This mirrors the algorithm shown in the lecture notes
            if p2.has_skip() and p2.skip_ptr <= p1:
                while p2.has_skip() and p2.skip_ptr <= p1:
                    p2 = p2.skip_ptr
            else:
                p2 = p2.next
    return convert_posting_to_list(results)


def apply_or(pl1: PostingsList, pl2: PostingsList) -> PostingsList:
    """
    Apply the OR operation on term 1 and term 2 using low level operations.
    We assume that the posting list file for both term exists.
    Skip pointers are useless for or
    """
    p1 = p2 = 0  # Pointers to each posting list
    p1 = pl1.initialise_linked_list()
    p2 = pl2.initialise_linked_list()
    results = []
    while p1 is not None and p2 is not None:
        if p1 == p2:
            results.append(p1.value)
            p1 = p1.next
            p2 = p2.next
        elif p1 < p2:
            results.append(p1.value)
            p1 = p1.next
        else:
            results.append(p2.value)
            p2 = p2.next
    while p1 is not None:
        results.append(p1.value)
        p1 = p1.next
    while p2 is not None:
        results.append(p2.value)
        p2 = p2.next
    return convert_posting_to_list(results)


def apply_and_not(pl1: PostingsList, pl2: PostingsList) -> PostingsList:
    """
    Apply the term1 AND NOT term2
    term1: 1 2 3 4 5
    term2: 2 3
    """
    p1 = pl1.initialise_linked_list()
    p2 = pl2.initialise_linked_list()
    results = []
    while p1 is not None and p2 is not None:
        if p1 < p2:
            results.append(p1.value)
            p1 = p1.next
        elif p1 == p2:
            p1 = p1.next
            p2 = p2.next
        else:
            # Same reasoning as AND skip pointer
            # We still want to jump to equality so that we can advance p1
            if p2.has_skip() and p2.skip_ptr <= p1:
                while p2.has_skip() and p2.skip_ptr <= p1:
                    p2 = p2.skip_ptr
            else:
                p2 = p2.next
    while p1 is not None:
        results.append(p1.value)
        p1 = p1.next
    return convert_posting_to_list(results)


def apply_not(pl: PostingsList, universe: PostingsList) -> PostingsList:
    """Get all terms in universe AND NOT pl"""
    return apply_and_not(universe, pl)


"""
Evaluation methods
"""
# We are using a posting lists to avoid reading from disk the same query two times.
# One thing to note: we avoid mutating the list itself in our boolean operations and only add the value to the result
# list. This is to avoid subtle bugs that may arise.
posting_lists = {}


class Term:
    def __init__(self, term) -> None:
        self.term = term

    def evaluate(self, indexer: Indexer):
        if self.term not in posting_lists:
            posting_lists[self.term] = indexer.get_posting_list(self.term)
        return posting_lists[self.term]

    def __repr__(self):
        return str(self.term)


class Not:
    def __init__(self, term: Term) -> None:
        self.term = term

    def evaluate(self, indexer: Indexer):
        ans = apply_not(self.term.evaluate(indexer), posting_lists[UNIVERSE])
        return reapply_skip_pointers(ans)

    def __repr__(self):
        return f"Not( {self.term} )"


class And:
    def __init__(self, terms) -> None:
        self.terms = terms

    def evaluate(self, indexer: Indexer):
        res = [term.evaluate(indexer) for term in self.terms]
        res.sort(key=lambda x: len(x))
        ans = res[0]
        for i in range(1, len(res)):
            ans = apply_and(ans, res[i])
            # Need to reapply skip pointers at every iteration
            ans = reapply_skip_pointers(ans)
        ans = reapply_skip_pointers(ans)
        return ans

    def __repr__(self):
        return f"And( {self.terms} )"


class Or:
    def __init__(self, terms) -> None:
        self.terms = terms

    def evaluate(self, indexer: Indexer):
        res = [term.evaluate(indexer) for term in self.terms]
        ans = res[0]
        for i in range(1, len(res)):
            ans = apply_or(ans, res[i])
            ans = reapply_skip_pointers(ans)
        return ans

    def __repr__(self):
        return f"Or( {self.terms} )"


def opt_eval(indexer: Indexer, query: list[str]):
    # DOES NOT ACCOUNT FOR (AND NOT): ONLY AND, OR, NOT
    # Make sure to reset posting_lists everytime
    posting_lists.clear()
    posting_lists[UNIVERSE] = indexer.get_posting_list(UNIVERSE)
    stack = []
    for term in query:
        if term == "AND":
            a = stack.pop()
            b = stack.pop()
            if isinstance(a, And) and isinstance(b, And):
                stack.append(And(a.terms + b.terms))
            elif isinstance(a, And):
                a.terms.append(b)
                stack.append(a)
            elif isinstance(b, And):
                b.terms.append(a)
                stack.append(b)
            else:
                stack.append(And([a, b]))
        elif term == "OR":
            a = stack.pop()
            b = stack.pop()
            if isinstance(a, Or) and isinstance(b, Or):
                stack.append(Or(a.terms + b.terms))
            elif isinstance(a, Or):
                a.terms.append(b)
                stack.append(a)
            elif isinstance(b, Or):
                b.terms.append(a)
                stack.append(b)
            else:
                stack.append(Or([a, b]))
        elif term == "NOT":
            top = stack.pop()
            stack.append(Not(top))
        else:  # regular term
            stack.append(Term(term))
    ans = stack[0].evaluate(indexer)
    ans = [str(posting.value) for posting in ans.plist]
    return " ".join(ans)


def naive_evaluation(indexer: Indexer, query: list[str]):
    """
    The most baseline evaluation that operates according to Shunting.
    This is thus therefore highly likely to be correct.
    """

    def get_posting_list(term):
        if isinstance(term, PostingsList):
            return term
        elif isinstance(term, str):
            return indexer.get_posting_list(term)
        else:
            raise ValueError("Invalid term")

    operand_stack = []
    for token in query:
        if token in ["AND", "OR", "AND NOT"]:
            s1 = operand_stack.pop()
            pl1 = get_posting_list(s1)

            s2 = operand_stack.pop()
            pl2 = get_posting_list(s2)

            if token == "AND":
                results = apply_and(pl1, pl2)
            elif token == "OR":
                results = apply_or(pl1, pl2)
            else:  # token == "AND NOT":
                # Note the swap on pl1 and pl2
                # The term at the top of the stack is the one that should be negated
                results = apply_and_not(pl2, pl1)
            results = reapply_skip_pointers(results)
            operand_stack.append(results)
        elif token in ["NOT"]:
            s = operand_stack.pop()
            pl = get_posting_list(s)
            universe = indexer.get_posting_list(UNIVERSE)
            results = apply_and_not(universe, pl)
            results = reapply_skip_pointers(results)
            operand_stack.append(results)
        else:
            operand_stack.append(token)
    assert len(operand_stack) == 1
    # Handle the unary case which has no optimisation possible
    if isinstance(operand_stack[0], str):
        operand_stack = [indexer.get_posting_list(operand_stack[0])]
    results = " ".join([str(posting.value) for posting in operand_stack[0]])
    return results


def naive_search(query: str, indexer: Indexer, stemmer: nltk.stem.PorterStemmer) -> str:
    splitted = split(query, stemmer)
    if splitted is None:
        return ""
    query_list = shunting(splitted)
    return naive_evaluation(indexer, query_list)


def search(query: str, indexer: Indexer, stemmer: nltk.stem.PorterStemmer) -> str:
    splitted = split(query, stemmer)
    if splitted is None:
        return ""
    query_list = shunting(splitted)
    return opt_eval(indexer, query_list)


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print("running search on the queries...")
    indexer = Indexer(dict_file, postings_file)
    stemmer = nltk.stem.PorterStemmer()
    indexer.load()
    with open(results_file, "w") as outf, open(queries_file, "r") as inf:
        num_queries = 0
        while True:
            query = inf.readline().strip()
            print(f"{num_queries}: OG Query is : " + query)
            if not query:
                break
            results = search(query, indexer, stemmer)
            # results = naive_search(query, indexer, stemmer)
            outf.write(results + "\n")
            num_queries += 1
        print(f"Handled {num_queries} queries")


def evaluate_runtime(
    dict_file, postings_file, queries_file, search_fn, num_iterations=10
):
    indexer = Indexer(dict_file, postings_file)
    stemmer = nltk.stem.PorterStemmer()
    indexer.load()
    total_time_taken = 0
    for _ in range(num_iterations):
        with open(queries_file, "r") as qf:
            start = time.time()
            while True:
                query = qf.readline().strip()
                if not query:
                    break
                _ = search_fn(query, indexer, stemmer)
            elapsedTime = time.time() - start
            total_time_taken += elapsedTime
    average_time_taken = total_time_taken / num_iterations
    print(f"Took {average_time_taken} seconds on average")


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "d:p:q:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == "-d":
        dictionary_file = a
    elif o == "-p":
        postings_file = a
    elif o == "-q":
        file_of_queries = a
    elif o == "-o":
        file_of_output = a
    else:
        assert False, "unhandled option"

if (
    dictionary_file == None
    or postings_file == None
    or file_of_queries == None
    or file_of_output == None
):
    usage()
    sys.exit(2)

# run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
print("Evaluating 'optimal' search")
evaluate_runtime(dictionary_file, postings_file, file_of_queries, search_fn=search)
print("Evaluating 'naive' search")
evaluate_runtime(
    dictionary_file, postings_file, file_of_queries, search_fn=naive_search
)
