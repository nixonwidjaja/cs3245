# Use multi-processing to quickly compute and cache the tokens of the dataset's
# "content" field.

import csv
import itertools
import math
import multiprocessing
import os
import time

from dataset import DataElement, Dataset
from preprocessor import Preprocessor


def process_element(element: DataElement) -> tuple[int, list[str]]:
    return element["document_id"], list(Preprocessor.to_token_stream(element["content"]))


if __name__ == "__main__":
    assert multiprocessing.cpu_count() > 1, f"You only have 1 CPU core, which can't support multi-processing."  # fmt:skip

    start_time = time.time()

    num_processes = math.floor(multiprocessing.cpu_count() * 0.8)
    print(f"Computing cache using {num_processes} processes (80% of avaliable CPU cores).")

    dataset = list(Dataset.load_dataset_stream("dataset.csv"))

    with multiprocessing.Pool(processes=num_processes) as pool:
        tokens_list = pool.map(process_element, dataset)

    os.makedirs(os.path.dirname(Dataset.CACHE_FILE_PATH), exist_ok=True)
    with open(Dataset.CACHE_FILE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for doc_id, tokens in tokens_list:
            writer.writerow(itertools.chain([doc_id], tokens))

    end_time = time.time()
    print(
        f"Multiprocessing ({num_processes} processes) tokenization time: {end_time - start_time:.1f}s"
    )
