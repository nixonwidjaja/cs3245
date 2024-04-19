import math
from collections import defaultdict

import nltk
from indexer import Indexer


class Scorer:
    """Handles computing query weights, documents' ranking scores. and
    performing Relevance-Feedback.
    """

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer
        self.query_weight: defaultdict[str, float] = defaultdict(lambda: 0.0)

    def init_term_weights(self, query_tokens: list[str]) -> None:
        """Set query weights to that of `query_tokens`."""
        N = self.indexer.num_docs
        tf_dict = nltk.FreqDist(query_tokens)
        for term, tf in tf_dict.items():
            df = self.indexer.get_df(term)

            # Ignore terms that don't appear in any docs.
            if df == 0:
                continue

            self.query_weight[term] = (1 + math.log10(tf)) * math.log10(N / df)

    def get_doc_scores(self) -> defaultdict[int, float]:
        """Compute documents' scores using current query weights."""
        scores: dict[int, float] = defaultdict(lambda: 0.0)
        for term, query_weight in self.query_weight.items():
            df, postings_list = self.indexer.get_term_data(term)

            for doc_id, doc_weight in postings_list:
                scores[doc_id] += doc_weight * query_weight

        return scores
