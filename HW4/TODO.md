# Setup
- If u don't have the dataset, u can download it by running this in Bash/Git Bash:
  ```bash
  bash download_dataset.sh
  ```
- Precompute the tokens (for local testing) by either:
    - for multi-processing, run `compute_cache.DONT_SUBMIT.py` and hope u have enough RAM
    - else, run
        ```py
        from dataset import Dataset
        Dataset.get_tokenized_content_stream("dataset.csv", save_cache=True)
        ```


# Things done:
- Use HW3 ranking system
- Change dictionary.txt / postings.txt encoding to store pickled objects
- Implement caching for the tokenized "content" field of the dataset
- Replace stemming -> lemmatization
- Query expansion via WordNet
- Pseudo Relevance Feedback (for both relevant and irrelevant docs)


# Things tested
- Stemming vs lemmatization vs lemmatization with POS tagging
    - lemmatization with POS tagging takes too long to index (~1h 30mins) and also performed worse than w/o POS
    - lemmatization w/o POS tagging performed better than the other 2, and also performed better with query-expansion which works with lemmas as WordNet works with lemmas
- Pseudo Relevance Feedback is kind of a mixed bag. But it seems to give better results when used modestly, and with only 1 iteration`
- Query expansion is definitely needed as, for example, Query Q1 is "quiet phone call" but the given relevant docs only has "silent telephone" which don't match the query. So the synonyms need to be searched too


# TODO
- Add positional index to postings
- Implement phrase searching for both boolean queries and for keyword-proximity seaching
- Somehow use the zones/fields? IDK how tho