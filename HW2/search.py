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
    2-word terms such as 'bunny balls' should be counted as invalid.
    Returns None if invalid
    """
    q = re.sub(r"[(]", "( ", q)
    q = re.sub(r"[)]", " )", q)
    # Split each query into words
    tokens = q.split()
    new_tokens = []
    # To count the number of consecutive regular terms
    regular_term_count = 0
    # Reject if operators located at invalid positions
    if tokens[0] in ["AND", "OR"] or tokens[-1] in ["AND", "OR", "NOT"]:
        return None
    for token in tokens:
        if token.upper() in ["AND", "OR", "NOT", "(", ")"]:
            # If token is operator, ensure uppercase and reset consecutive regular terms count
            regular_term_count = 0
            new_tokens.append(token.upper())
        else:
            # If detected 2 consecutive regular terms, reject the query
            regular_term_count += 1
            if regular_term_count > 1:
                return None
            # Use stemming to match the preprocessing of index
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

    # Set operator precedence
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


def opt_shunting(tokens) -> list[str]:
    """
    Given a sequence of tokens and using Shunting Yard algorithm,
    return a combination of Term, Not, And, Or objects to be evaluated.
    """
    if not tokens:
        return
    operator_stack = []
    term_stack = []

    # Set operator precedence
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
        if token in operators:
            terms = []
            # While topmost operator precedence > current precedence, group previous Not() and And()
            while operator_stack and PRECEDENCE[operator_stack[-1]] > PRECEDENCE[token]:
                last_operator = operator_stack.pop()
                # Form a Not(term) and append to term_stack
                if last_operator == "NOT":
                    term_stack.append(Not(term_stack.pop()))
                    continue
                terms.append(term_stack.pop())
            if terms:
                # Form an And([terms]) and append to term_stack
                terms.append(term_stack.pop())
                term_stack.append(And(terms))
            # Append current operator to operator_stack
            operator_stack.append(token)
        elif token == "(":
            # If current token is "(", find the next ")" and do opt_shunting for tokens inside ()
            right_parenth_idx = tokens[i:].index(")") + i
            result = opt_shunting(tokens[i + 1 : right_parenth_idx])
            # Move pointer i to ")"
            i = right_parenth_idx
            # Append the resulting term to term_stack
            term_stack.append(result)
        else:  # regular terms
            term_stack.append(Term(token))
        i += 1
    terms = []
    now = None
    # After all query is inside the stacks, process the stacks
    while operator_stack:
        last_operator = operator_stack.pop()
        # Form a Not(term) and append to term_stack
        if last_operator == "NOT":
            term_stack.append(Not(term_stack.pop()))
            continue
        if last_operator != now:
            if terms:
                terms.append(term_stack.pop())
                # Form an And([terms]) and append to term_stack
                if now == "AND":
                    term_stack.append(And(terms))
                # Form an Or([terms]) and append to term_stack
                if now == "OR":
                    term_stack.append(Or(terms))
                terms = []
            now = last_operator
        terms.append(term_stack.pop())
    # Process the final term
    if terms:
        terms.append(term_stack.pop())
        if now == "AND":
            term_stack.append(And(terms))
        if now == "OR":
            term_stack.append(Or(terms))
    return term_stack[0]


"""
BOOLEAN OPERATORS
"""


def convert_posting_to_list(result: list[int]) -> PostingsList:
    """Converts list of doc IDs to PostingsList"""
    pl = PostingsList()
    pl.plist = [Posting(docId) for docId in result]
    return pl


def reapply_skip_pointers(pl: PostingsList) -> PostingsList:
    """
    When merging two posting lists, their old skip pointers will
    no longer be valid as it is possible we take a posting from one
    posting list that points to an invalid index!
    We will apply a linear time reapplication of the skip pointers.
    There is no need to sort because we process posting lists in ascending order.
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
    Make use of skip pointers whenever possible.
    Returns the intersection of term 1 and term 2's posting lists.
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
    Skip pointers are useless for OR.
    Returns the union of term 1 and term 2's posting lists.
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
    result: 1 4 5
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
    """Apply the NOT operation to a posting list by applying UNIVERSE AND NOT term"""
    return apply_and_not(universe, pl)


"""
Evaluation methods
"""


class Term:
    """
    Term abstraction that evaluates to the posting list of the term.
    """

    def __init__(self, term) -> None:
        self.term = term

    def evaluate(self, indexer: Indexer):
        return indexer.get_posting_list(self.term)

    def __repr__(self):
        return str(self.term)


class Not:
    """
    Not abstraction of Not(Term) that evaluates to the posting list of
    the result of 'NOT term'.
    """

    def __init__(self, term: Term) -> None:
        self.term = term

    def evaluate(self, indexer: Indexer):
        ans = apply_not(self.term.evaluate(indexer), indexer.get_posting_list(UNIVERSE))
        return reapply_skip_pointers(ans)

    def __repr__(self):
        return f"Not( {self.term} )"


class And:
    """
    And abstraction of And([terms]) that evaluates to the posting list of
    the result of 'terms[0] AND terms[1] AND ... AND terms[n-1]'.
    """

    def __init__(self, terms) -> None:
        self.terms = terms

    def evaluate(self, indexer: Indexer):
        res = [term.evaluate(indexer) for term in self.terms]
        # Sort the terms in the order of increasing posting list length to optimise
        res.sort(key=lambda x: len(x))
        ans = res[0]
        for i in range(1, len(res)):
            ans = apply_and(ans, res[i])
            # Need to reapply skip pointers at every iteration
            ans = reapply_skip_pointers(ans)
        return ans

    def __repr__(self):
        return f"And( {self.terms} )"


class Or:
    """
    Or abstraction of Or([terms]) that evaluates to the posting list of
    the result of 'terms[0] OR terms[1] OR ... OR terms[n-1]'.
    """

    def __init__(self, terms) -> None:
        self.terms = terms

    def evaluate(self, indexer: Indexer):
        res = [term.evaluate(indexer) for term in self.terms]
        ans = res[0]
        for i in range(1, len(res)):
            ans = apply_or(ans, res[i])
            # Need to reapply skip pointers at every iteration
            ans = reapply_skip_pointers(ans)
        return ans

    def __repr__(self):
        return f"Or( {self.terms} )"


def naive_evaluation(indexer: Indexer, query: list[str]):
    """
    The most baseline evaluation that operates according to Shunting Yard.
    Query is assumed to be in postfix notation.
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
            # If current token is an operator, get 2 topmost token and apply the operator
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
        elif token == "NOT":
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
    # Split the query into operators and regular terms, stem regular terms
    splitted = split(query, stemmer)
    # If invalid query, reject and return ""
    if splitted is None:
        return ""
    # Apply Shunting Yard to splitted query to get postfix notation
    query_list = shunting(splitted)
    # Apply naive evaluation to postfix query
    return naive_evaluation(indexer, query_list)


def opt_search(query: str, indexer: Indexer, stemmer: nltk.stem.PorterStemmer) -> str:
    # Split the query into operators and regular terms, stem regular terms
    splitted = split(query, stemmer)
    # If invalid query, reject and return ""
    if splitted is None:
        return ""
    # Apply optimised Shunting Yard and evaluate the result
    ans = opt_shunting(splitted).evaluate(indexer)
    # Convert posting list to string result
    ans = [str(posting.value) for posting in ans.plist]
    return " ".join(ans)


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print("running search on the queries...")
    # Initialize indexer and PorterStemmer
    indexer = Indexer(dict_file, postings_file)
    stemmer = nltk.stem.PorterStemmer()
    # Load dictionary mapping of token to file pointer from dict_file
    indexer.load()
    with open(results_file, "w") as outf, open(queries_file, "r") as inf:
        num_queries = 0
        while True:
            # Read a single line of query
            query = inf.readline().strip()
            if not query:
                break
            # Perform optimised search
            results = opt_search(query, indexer, stemmer)
            # results = naive_search(query, indexer, stemmer)
            # Write the result to results_file
            outf.write(results + "\n")
            num_queries += 1
        print(f"Handled {num_queries} queries")


def evaluate_runtime(
    dict_file, postings_file, queries_file, search_fn, num_iterations=10
):
    """
    Calculate average runtime to get a performance benchmark.
    """
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

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
print("Evaluating 'naive' search")
evaluate_runtime(
    dictionary_file, postings_file, file_of_queries, search_fn=naive_search
)
print("Evaluating 'optimised' search")
evaluate_runtime(dictionary_file, postings_file, file_of_queries, search_fn=opt_search)
