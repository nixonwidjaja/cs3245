# Use multi-processing to quickly compute and cache the tokens of the dataset's
# "content" field.

import multiprocessing
import pickle
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

    with open(Dataset.CACHE_FILE_PATH, "wb") as f:
        pickle.dump(tokens_list, f)

    end_time = time.time()
    print(f"Tokenization time: {end_time - start_time:.2f} seconds")
