# Use multi-processing to quickly compute and cache the tokens of the dataset's
# "content" field.

import csv
import itertools
import multiprocessing
import os
import time

from dataset import DataElement, Dataset
from preprocessor import Preprocessor

NUM_PROCESSES = 30


def process_element(element: DataElement) -> tuple[int, list[str]]:
    return element["document_id"], list(Preprocessor.to_token_stream(element["content"]))


if __name__ == "__main__":
    assert multiprocessing.cpu_count() >= NUM_PROCESSES, f"You have less than NUM_PROCESSES={NUM_PROCESSES} number of cores."  # fmt:skip
    dataset = list(Dataset.load_dataset_stream("dataset.csv"))

    start_time = time.time()

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        tokens_list = pool.map(process_element, dataset)

    os.makedirs(os.path.dirname(Dataset.CACHE_FILE_PATH), exist_ok=True)
    with open(Dataset.CACHE_FILE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for doc_id, tokens in tokens_list:
            writer.writerow(itertools.chain([str(doc_id)], tokens))

    end_time = time.time()
    print(
        f"Multiprocessing ({NUM_PROCESSES} processes) tokenization time: {end_time - start_time:.1f}s"
    )
