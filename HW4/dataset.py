import csv
import sys
from typing import Iterator, TypedDict


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
