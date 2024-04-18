from linked_list import LinkedList


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

        Format of dictionary file, where the doc-lengths are seperated from the
        terms' metadata by a empty line:
        ```
        TERM DF OFFSET SIZE   # eg. bahia 4 0 37
        TERM DF OFFSET SIZE
        ...

        DOC_ID NORMALIZED_DOC_LENGTH   # eg. 1 18.743376382392842
        DOC_ID NORMALIZED_DOC_LENGTH
        ...
        ```

        Returns:
            tuple[dict[str, tuple[int, int, int]], dict[int, float]]: \
                `(term_metadata, doc_norm_lengths)`.
        """
        term_metadata: dict[str, tuple[int, int, int]] = {}
        doc_norm_lengths: dict[int, float] = {}

        with open(dict_file_path, "r") as f:
            # Read the terms' metadata (ie. DF, offset, size) first.
            for line in f:
                if line == "\n":
                    break
                term, doc_freq, offset, size = line.rstrip("\n").split()
                term_metadata[term] = int(doc_freq), int(offset), int(size)

            # Then read the normalized doc lengths, which is separated by a
            # empty line from the terms' metadata.
            for line in f:
                if line == "\n":
                    break
                docid, norm_length = line.rstrip("\n").split()
                doc_norm_lengths[int(docid)] = float(norm_length)

        return term_metadata, doc_norm_lengths

    def get_term_data(self, term: str) -> tuple[int, LinkedList[tuple[int, int]]]:
        """Gets the term's DF (from dictionary file) and postings list (from
        postings file).

        The postings list is in the form:
        ```
        [(DOC_ID, TF), ...]
        ```

        Format of postings file:
        ```
        (DOC_ID,TF) (DOC_ID,TF) ...   # eg. (1,4) (11459,2) (11911,3) (13462,1)
        (DOC_ID,TF) (DOC_ID,TF) ...
        ...
        ```

        Returns:
            tuple[int, LinkedList[tuple[int, int]]]: `(df, postings_list)`
        """
        if term not in self.term_metadata:
            return 0, LinkedList.NULL_NODE

        df, offset, size = self.term_metadata[term]
        self.postings_file_io.seek(offset)
        raw_postings_str = self.postings_file_io.read(size).decode().rstrip()

        postings_list: list[tuple[int, int]] = []
        for s in raw_postings_str.split(" "):
            docid, term_freq = s.rstrip(")").lstrip("(").split(",")
            postings_list.append((int(docid), int(term_freq)))
        return df, LinkedList.from_list(postings_list)
