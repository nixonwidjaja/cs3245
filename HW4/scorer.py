import heapq
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

    def apply_relevance_feedback(
        self,
        *,
        alpha: float = 1,
        relevant_doc_ids: list[int] = [],
        beta: float = 0,
        irrelevant_doc_ids: list[int] = [],
        gamma: float = 0,
    ) -> None:
        """Update query weights via Relevance-Feedback.

        Args:
            alpha (float, optional): Scaling for previous query weights. Defaults to 1.
            relevant_doc_ids (list[int], optional): Relevant doc-IDs. Defaults to [].
            beta (float, optional): Scaling for relevant docs. Defaults to 0.
            irrelevant_doc_ids (list[int], optional): Irrelevant doc-IDs. Defaults to [].
            gamma (float, optional): Scaling for irrelevant docs. Defaults to 0.
        """
        # Scale previous query weights.
        for term in self.query_weight.keys():
            self.query_weight[term] *= alpha

        # For relevant docs.
        if beta and relevant_doc_ids:
            num_relevant = len(relevant_doc_ids)
            summed_vector = nltk.FreqDist()
            for doc_id in relevant_doc_ids:
                doc_vector = self.indexer.get_doc_vector(doc_id)
                summed_vector.update(doc_vector)

            for term in summed_vector.keys():
                self.query_weight[term] += beta * (summed_vector[term] / num_relevant)

        # For irrelevant docs.
        if gamma and irrelevant_doc_ids:
            num_irrelevant = len(irrelevant_doc_ids)
            summed_vector = nltk.FreqDist()
            for doc_id in irrelevant_doc_ids:
                doc_vector = self.indexer.get_doc_vector(doc_id)
                summed_vector.update(doc_vector)

            for term in summed_vector.keys():
                self.query_weight[term] -= gamma * (summed_vector[term] / num_irrelevant)

    def apply_pseudo_relevance_feedback(
        self,
        *,
        alpha: float = 1,
        n_relevant: int = 0,
        beta: float = 0,
    ) -> None:
        """Update query weights by doing Relevance-Feedback using the top
        scoring `n_relevant` num. of documents.

        Args:
            alpha (float, optional): Scaling for previous query weights. Defaults to 1.
            n_relevant (int, optional): Num. of top scoring documents to use for \
                Relevance Feedback. Defaults to 0.
            beta (float, optional): Scaling for relevant docs. Defaults to 0.
        """
        scores = self.get_doc_scores()
        relevant_doc_ids: list[int] = []

        if beta and n_relevant:
            relevant_doc_ids = heapq.nlargest(
                n_relevant,
                scores.keys(),
                key=lambda doc_id: scores[doc_id],
            )

        self.apply_relevance_feedback(
            alpha=alpha,
            relevant_doc_ids=relevant_doc_ids,
            beta=beta,
        )
