from typing import Iterator

from indexer import Indexer


def search_permutes(n: int) -> Iterator[tuple[list[int], list[int]]]:
    """Gets permutations of which token indices are in a phrase, and which are
    in single keywords.

    For example:
    For the tokens: [t0, t1, t2, t3, t4]
    `([1, 2, 3], [0, 4])` means:
    `"t1 t2 t3" AND "t0" AND "t4"`

    Yields:
        `(indices_in_phrase, indices_not_in_phrase)`
    """
    full_list = list(range(n))
    yield full_list, []

    for length in range(n - 1, 1, -1):
        for start in range(n - length + 1):
            yield full_list[start : start + length], full_list[:start] + full_list[start + length :]
            yield full_list[start : start + length], []

    yield [], full_list


def apply_permutation(
    elements: list[str],
    permutation: tuple[list[int], list[int]],
) -> list[list[str]]:
    """Applies the permutation from `serach_permutes` to get a list of list of
    tokens.

    For example:
    returning `[["t1", "t2"], ["t3"], ["t4"]]` means:
    "t1 t2" AND "t3" AND "t4"
    """
    phrase_indices, other_indices = permutation
    if len(phrase_indices) == 0:
        return [[elements[i]] for i in other_indices]
    return [[elements[i] for i in phrase_indices], *([elements[i]] for i in other_indices)]


def merge_consecutive_postings_list(
    postings_list_1: list[tuple[int, list[int]]],
    postings_list_2: list[tuple[int, list[int]]],
) -> Iterator[tuple[int, list[int]]]:
    """Merge 2 postings list, where `postings_list_2` is for the term right
    after that of `postings_list_1`.
    
    For example, for the query "phone call", `postings_list_1` will be for
    "phone", and `postings_list_2` will be for "call".

    The postings lists format is [(doc_id, positional_indices), ...].

    Yields:
        Iterator[tuple[int, list[int]]]: `(doc_id, positional_indices_of_list_2)`, \
            where `positional_indices_of_list_2` is the positional indices of the \
            matching position, specifically from the perspective of `postings_list_2`.
    """
    iter_1 = iter(postings_list_1)
    iter_2 = iter(postings_list_2)

    posting_1 = next(iter_1, None)
    posting_2 = next(iter_2, None)

    while posting_1 and posting_2:
        doc_id_1, indices_1 = posting_1
        doc_id_2, indices_2 = posting_2

        if doc_id_1 < doc_id_2:
            posting_1 = next(iter_1, None)
        elif doc_id_1 > doc_id_2:
            posting_2 = next(iter_2, None)
        else:
            merged_indices = []
            iter_indices_1 = iter(indices_1)
            iter_indices_2 = iter(indices_2)

            index_1 = next(iter_indices_1, None)
            index_2 = next(iter_indices_2, None)

            while index_1 is not None and index_2 is not None:
                if index_1 + 1 == index_2:
                    merged_indices.append(index_2)
                    index_1 = next(iter_indices_1, None)
                    index_2 = next(iter_indices_2, None)
                elif index_1 < index_2:
                    index_1 = next(iter_indices_1, None)
                else:
                    index_2 = next(iter_indices_2, None)

            if merged_indices:
                yield (doc_id_1, merged_indices)

            posting_1 = next(iter_1, None)
            posting_2 = next(iter_2, None)


def merge_postings_list(
    postings_list_1: list[tuple[int, list[int]]],
    postings_list_2: list[tuple[int, int]],
) -> Iterator[tuple[int, list[int]]]:
    """Merge 2 postings list, where both lists are AND operated.
    
    For example, for the query "phone AND call", `postings_list_1` will be for
    "phone", and `postings_list_2` will be for "call".

    The postings lists format is [(doc_id, positional_indices), ...].

    Yields:
        Iterator[tuple[int, tuple[int, int]]]: `(doc_id, (tf_1, tf_2))`, where \
        `tf_1` is the term-frequency for `postings_list_1` for `doc_id`, and \
        `tf_2` is that for `postings_list_2`.
    """
    iter_1 = iter(postings_list_1)
    iter_2 = iter(postings_list_2)

    posting_1 = next(iter_1, None)
    posting_2 = next(iter_2, None)

    while posting_1 and posting_2:
        doc_id_1, tf_1 = posting_1
        doc_id_2, tf_2 = posting_2

        if doc_id_1 < doc_id_2:
            posting_1 = next(iter_1, None)
        elif doc_id_1 > doc_id_2:
            posting_2 = next(iter_2, None)
        else:
            yield doc_id_1, (tf_1 + [tf_2])
            posting_1 = next(iter_1, None)
            posting_2 = next(iter_2, None)


query: list[list[str]] = [["term2", "term3"], ["term0"], ["term1"]]
"""Represents the query: "term2 term3" AND "term0" AND "term1"."""


def solve_query(indexer: Indexer, query: list[list[str]]) -> list[tuple[int, list[int]]]:
    postings_tfs: list[list[tuple[int, int]]] = []
    for phrase in query:
        if len(phrase) == 1:
            postings_tfs.append(
                [
                    (doc_id, len(pos_indices))
                    for doc_id, pos_indices in indexer.get_postings_list(phrase[0])
                ]
            )
        else:
            tf_list = indexer.get_postings_list(phrase[0])
            for term in phrase[1:]:
                next_postings_list = indexer.get_postings_list(term)
                tf_list = list(merge_consecutive_postings_list(tf_list, next_postings_list))
            postings_tfs.append([(doc_id, len(pos_indices)) for doc_id, pos_indices in tf_list])

    result = [(doc_id, [tf]) for doc_id, tf in postings_tfs[0]]
    for tf_list in postings_tfs[1:]:
        result = list(merge_postings_list(result, tf_list))

    return result
