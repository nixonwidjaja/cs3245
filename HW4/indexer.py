import pickle


class Indexer:
    """Handles reading the dictionary and postings files."""

    def __init__(self, dict_file_path: str, postings_file_path: str) -> None:
        """
        Args:
            dict_file_path (str): Path to file containing the DF, offset and size for each term, \
                and all the normalized document lengths.
            postings_file_path (str): Path to file containing all the postings lists.
        """
        self.postings_file_io = open(postings_file_path, "rb")
        self.term_metadata, self.doc_norm_lengths = self._load_data_from_dict_file(dict_file_path)
        self.num_docs: int = len(self.doc_norm_lengths)

    def __enter__(self) -> "Indexer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close any opened file IOs."""
        self.postings_file_io.close()

    @staticmethod
    def _load_data_from_dict_file(
        dict_file_path: str,
    ) -> tuple[dict[str, tuple[int, int, int]], dict[int, float]]:
        """Loads the data from the dictionary file into memory.

        Specifically, it loads:
        - `term_metadata`: DF, offset, size for each term
          (where offset and size are used to seek/read from the postings file).
        - `doc_norm_lengths`: Cosine-normalization lengths for each doc-ID.

        Format of dictionary file:
        Bytes of the pickled `(term_metadata, doc_norm_lengths)` tuple.

        Returns:
            tuple[dict[str, tuple[int, int, int]], dict[int, float]]: \
                `(term_metadata, doc_norm_lengths)`.
        """
        with open(dict_file_path, "rb") as f:
            term_metadata, doc_norm_lengths = pickle.load(f)
            return term_metadata, doc_norm_lengths

    def get_term_data(self, term: str) -> tuple[int, list[tuple[int, int]]]:
        """Gets the term's DF (from dictionary file) and postings list (from
        postings file).

        The postings list is in the form:
        ```
        [(DOC_ID, TF), ...]
        ```

        Format of postings file:
        Bytes of all pickled postings lists.

        Returns:
            tuple[int, list[tuple[int, int]]]: `(df, postings_list)`
        """
        if term not in self.term_metadata:
            return 0, []

        df, offset, size = self.term_metadata[term]
        self.postings_file_io.seek(offset)
        postings_list = pickle.loads(self.postings_file_io.read(size))
        return df, postings_list
