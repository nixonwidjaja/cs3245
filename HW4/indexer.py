import pickle

from struct import pack, unpack

def __encode(number):
    b = []
    while True:
        b.insert(0, number % 128)
        if number < 128:
            break
        number = number // 128
    b[-1] += 128
    return pack('%dB' % len(b), *b)


def gap_encode(numbers) -> list[int]:
    gaps = [numbers[0]]
    for i in range(1, len(numbers)):
        gaps.append(numbers[i] - numbers[i-1])
    return gaps


def vb_encode(numbers):
    bytes_list = []
    for number in numbers:
        bytes_list.append(__encode(number))
    return b"".join(bytes_list)


def vb_decode(bytestream):
    n = 0
    numbers = []
    bytestream = unpack('%dB' % len(bytestream), bytestream)
    for byte in bytestream:
        if byte < 128:
            n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0
    return numbers

class Indexer:
    """Handles reading the dictionary and postings files."""

    def __init__(self, dict_file_path: str, postings_file_path: str, use_compression: bool = False) -> None:
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
        self.use_compression = use_compression

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
        if self.use_compression:
            # We now need to reconstruct the postings_list in the expected format from the gap and vb encoding
            vb_gaps, doc_weights = postings_list
            gaps = vb_decode(vb_gaps)
            pl = [(gaps[0], doc_weights[0])]
            for i in range(1, len(gaps)):
                pl.append((pl[i-1][0] + gaps[i], doc_weights[i]))
            return df, pl
        
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
