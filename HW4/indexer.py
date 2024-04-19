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
        self.term_metadata, self.doc_metadata = self._load_data_from_dict_file(dict_file_path)
        self.num_docs: int = len(self.doc_metadata)
        self.doc_ids = list(self.doc_metadata.keys())

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
    ) -> tuple[dict[str, tuple[int, int, int]], dict[int, tuple[int, int]]]:
        """Loads the data from the dictionary file into memory.

        Specifically, it loads:
        - `term_metadata`: DF, offset, size for each term
          (where offset and size are used to seek/read the postings list from the postings file).
        - `doc_metadata`: Cosine-normalization length, offset, size for each doc-ID
          (where offset and size are used to seek/read the doc vector from the postings file).

        Returns:
            tuple[dict[str, tuple[int, int, int]], dict[int, tuple[int, int]]]: \
                `(term_metadata, doc_metadata)`.
        """
        with open(dict_file_path, "rb") as f:
            term_metadata, doc_metadata = pickle.load(f)
            return term_metadata, doc_metadata

    def get_term_data(self, term: str) -> tuple[int, list[tuple[int, int]]]:
        """Returns `(df, postings_list)`, the term's DF (from dictionary file)
        and postings list (from postings file).

        Returns:
            tuple[int, list[tuple[int, int]]]: `(df, postings_list)`
        """
        if term not in self.term_metadata:
            return 0, []

        df, offset, size = self.term_metadata[term]
        self.postings_file_io.seek(offset)
        postings_list = pickle.loads(self.postings_file_io.read(size))
        return df, postings_list

    def get_df(self, term: str) -> int:
        """Gets a term's DF (from dictionary file)."""
        if term not in self.term_metadata:
            return 0

        df, *_ = self.term_metadata[term]
        return df

    def get_doc_vector(self, doc_id: int) -> dict[str, float]:
        """Gets a document's weight vector (from postings file)."""
        offset, size = self.doc_metadata[doc_id]
        self.postings_file_io.seek(offset)
        doc_vector = pickle.loads(self.postings_file_io.read(size))
        return doc_vector
