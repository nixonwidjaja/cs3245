import re

query = "bill OR Gates AND (vista OR XP) AND NOT mac"
expected = "bill Gates OR "


class AND:
    def __init__(self, queries):
        self.queries = queries
        
    def add(self, query):
        if isinstance(query, list):
            self.queries.extend(query)
        else:
            self.queries.append(query)
        return self
        
    def __repr__(self):
        return "AND " + str(self.queries) + " "
    
class OR:
    def __init__(self, queries):
        self.queries = queries
        
    def add(self, query):
        self.queries.append(query)
        return self

    def __repr__(self):
        return "OR " + str(self.queries) + " "


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
    """Evaluate OR expressions separately, AND expressions together"""
    # Add whitespaces so that we can split by parentheses properly
    q = re.sub(r'[(]', "( ", q)
    q = re.sub(r'[)]', " )", q)
    tokens = q.split()
    return tokens
    
def parse(tokens):
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
            result = parse(tokens[i+1:right_parenth_idx])
            i = right_parenth_idx
            for t in result:
                result_stack.append(t)
        else:
            result_stack.append(token)
        i += 1
    flush()
    return result_stack
    

def transform_ast(query):
    """Assumes that it is in the postfix notation.
    Uses a simple optimization.
    We treat all ORs separately
    """
    print(query)
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


def optimize(ast):
    print(ast)

# print(evaluate(query))

# query = "(madding OR crowd) AND (ignoble OR strife) AND (killed OR slain)"
# print(transform_ast(parse(split(query))))

# frequency = {
#     "brutus": 7,
#     "caesar": 8,
#     "calpurnia": 2,
# }
# query = "bill OR Gates AND (vista OR XP) AND NOT mac"
# optimize(transform_ast(parse(split(query))))

def parse_query(query):
    return transform_ast(parse(split(query)))

def print_parse(query):
    print(query)
    query = parse_query(query)
    print(query)
    
def evaluate(query, freq):
    """Recursive evaluate
    AND can have n terms. Recursive estimate the size of each and order in ascending order.
    OR can only have 2 terms
    """
    print(query)
    print(f"Evaluate on {query}")
    if isinstance(query, AND):
        costs = [(evaluate(subquery, freq), subquery) for subquery in query.queries]
        optimised_query = [subquery for (cost, subquery) in sorted(costs, key=lambda x: x[0])]
        new_optimised_query = AND([optimised_query[0], optimised_query[1]])
        for i in range(2, len(optimised_query)):
            new_optimised_query = AND([new_optimised_query, optimised_query[i]])
        print(new_optimised_query)
    elif isinstance(query, OR):
        ...
    elif isinstance(query, AND_NOT):
        ...
    elif isinstance(query, NOT):
        ...
    else:
        return freq[query]
    return new_optimised_query

# with open("sanity-queries.txt", "r") as inf, open("sanity-queries-transformed.txt", "w") as outf:
#     for line in inf.readlines():
#         line = line.strip()
#         print(transform_ast(parse(split(line))))
#         outf.write(line + "\n")
#         outf.write(str(transform_ast(parse(split(line)))) + "\n")


freq = {
    "Caesar": 8,
    "Calpurnia": 2,
    "Brutus": 7
}

q = "Caesar AND Brutus AND Calpurnia"
print_parse(q)
evaluate(parse_query(q), freq)

freq = {
    "eyes": 213312,
    "kaleidoscope": 87009,
    "marmalade": 107913,
    "skies": 271658,
    "tangerine": 46653,
    "trees": 316812
}
# q = "(tangerine OR trees) AND (marmalade OR skies) AND (kaleidoscope OR eyes)"
# print_parse(q)

q = "a OR b OR c AND d"
print_parse(q)


"""
Query optimisation
AND(...,...) recursively optimise each query
"""

freq = {
    "Singapore": 300,
    "University": 700,
    "National": 1100,
    "NUS": 30,
    "Nanyang": 4,
    "Management": 1400,
    "the": 2500
}

q = "(Singapore AND University AND National) OR (NUS AND NOT NANYANG AND NOT Management)"
print_parse(q)