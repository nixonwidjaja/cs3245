import csv
import itertools
import os
import sys
from typing import Iterator, TypedDict

from preprocessor import Preprocessor
from tqdm.autonotebook import tqdm


class DataElement(TypedDict):
    document_id: int
    title: str
    content: str
    date_posted: str
    court: str


def _set_csv_limit_to_max():
    """Set CSV size limit to the max supported value."""
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10


class Dataset:
    """Handles the loading of dataset."""

    CACHE_FILE_PATH = f"cache/content_tokens.{Preprocessor.PREPROCESSING_MODE}.csv"

    NUM_DOCUMENTS = 17137
    """Total num. of documents in the dataset (doesn't include duplicate Doc-IDs)."""

    @staticmethod
    def load_dataset_stream(dataset_path: str) -> Iterator[DataElement]:
        """Yields each dataset row one-by-one as dictionaries.

        Each dictionary (ie. a dataset row) looks like:
        ```
        {
            "document_id": "246403",
            "title": "Yap Giau Beng Terence v Public Prosecutor [1998] SGHC 232",
            "content": "Yap Giau Beng Terence ...",
            "date_posted": "1998-07-08 00:00:00",
            "court": "SG High Court"
        }
        ```

        There are rows with duplicate Doc-IDs. These rows only differ by
        "court", and are handled by concatenating the court strings of all the
        duplicates.

        For example, Row 67 & 68 have Doc-ID 247336, with court `"HK Court of First Instance"`
        & `"HK High Court"` respectively. These 2 rows are combined into 1 dict with:
        `"court": "HK Court of First Instance HK High Court"`.
        """
        _set_csv_limit_to_max()

        with open(dataset_path, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip first line, which is the header.

            document_id, title, content, date_posted, court = next(reader)
            last_element: DataElement = {
                "document_id": int(document_id),
                "title": title,
                "content": content,
                "date_posted": date_posted,
                "court": court,
            }

            for document_id, title, content, date_posted, court in reader:
                document_id = int(document_id)

                # Handle duplicate rows with IDs (duplicates should only differ by "court").
                if document_id == last_element["document_id"]:
                    # Ensure only "court" differs
                    assert title == last_element["title"]
                    assert content == last_element["content"]
                    assert date_posted == last_element["date_posted"]

                    # Concat duplicate rows' courts.
                    last_element["court"] += " " + court
                    continue

                yield last_element
                last_element = {
                    "document_id": document_id,
                    "title": title,
                    "content": content,
                    "date_posted": date_posted,
                    "court": court,
                }

            yield last_element

    @staticmethod
    def get_tokenized_content_stream(
        dataset_path: str,
        save_cache: bool = False,
        validate_cache: bool = True,
    ) -> Iterator[tuple[int, list[str]]]:
        """Yields tuples of `(Doc-ID, token_list)`, one for each document, where
        `token_list` is the tokens of the document's `"content"`.

        If a precomputed cache of the tokens list exists, load the cache instead.

        Args:
            dataset_path (str): Path to the CSV dataset file.
            save_cache (bool, optional): Whether to save a cache of the computed tokens. \
                Defaults to False.
            validate_cache (bool, optional): Whether to validate the cached tokens list \
                (if it exists). Defaults to True.
        """
        # Load from cache if exists.
        if os.path.exists(Dataset.CACHE_FILE_PATH):
            print(f'Using cache at "{Dataset.CACHE_FILE_PATH}"')

            if validate_cache:
                dataset_stream = Dataset.load_dataset_stream(dataset_path)

            with open(Dataset.CACHE_FILE_PATH, newline="") as f:
                for i, (doc_id_str, *tokens) in enumerate(csv.reader(f)):
                    doc_id = int(doc_id_str)

                    if validate_cache:
                        element = next(dataset_stream)
                        if i % 500 == 0:
                            assert doc_id == element["document_id"]
                            assert tokens == list(Preprocessor.to_token_stream(element["content"]))

                    yield doc_id, tokens

            if validate_cache:
                # Assert that `dataset_stream` has exhausted.
                try:
                    next(dataset_stream)
                    raise AssertionError('"dataset_stream" iterator not exhausted.')
                except StopIteration:
                    pass

            return

        if save_cache:
            with open(Dataset.CACHE_FILE_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                for element in tqdm(
                    Dataset.load_dataset_stream(dataset_path), total=Dataset.NUM_DOCUMENTS
                ):
                    tokens = list(Preprocessor.to_token_stream(element["content"]))
                    yield element["document_id"], tokens
                    writer.writerow(itertools.chain([element["document_id"]], tokens))
            return

        for element in Dataset.load_dataset_stream(dataset_path):
            tokens = list(Preprocessor.to_token_stream(element["content"]))
            yield element["document_id"], tokens
