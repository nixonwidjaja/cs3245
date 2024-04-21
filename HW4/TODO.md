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

        for tokens in Dataset.get_tokenized_content_stream("dataset.csv", save_cache=True):
            pass
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
- Phrase searching (I only briefly tested, didn't push any code on it). \
  It gave very promising results, increasing the ranks of the given relevant docs from 200-400 down to <150. \
  But some problems that could arise:
  - Adding positional index to postings.txt increased its size from 652MB -> 1.1GB (ie. over the 800MB limit). So some compression needs to be done if we decide to do this.
  - Need a function to efficiently merge postings lists & their positional index. (ie. get the postings where the position index of 2nd term is +1 from the 1st term etc.)
  - Need a way to "loosen" the phrase searching. \
    eg. if "quiet phone call" yields too little results, loosen the query to "quiet" AND "phone call" etc.
  - Phrase searching doesn't work with Pseudo Relevance Feedback (PRF), so PRF needs should be removed when redesigning the search to do phrase searching


# TODO
- Write the README.txt. \
  Things needed in readme from [HW4 instructions](https://www.comp.nus.edu.sg/~cs3245/hw4-intelllex.html#:~:text=35%25%20Documentation.%20This,in%20the%20document.):
  - > 15% For your high level documentation, in your README document. This component comprises of an overview of your system, your system architecture, the techniques used to improve the retrieval performance, and the allocation of work to each of the individual members of the project team. In particular, describe the techniques you have implemented / experimented with. Discuss about the effects of those techniques on the retrieval performance with reference to some experimental results and analysis. If you have implemented two or more query expansion techniques for bonus marks, you should put all the information related to those techniques in BONUS.docx.
  - > 20% (Bonus marks) Exploration on query refinement. Describe the query refinement techniques you have implemented / experimented with. Discuss about the effects of those techniques on the retrieval performance with reference to some experimental results and analysis. The bonus marks will be awarded based on the number / correctness / complexity of the techniques implemented, as well as the amount / quality of the discussion in the document.
- Add positional index to postings
- Implement phrase searching for both boolean queries and for keyword-proximity seaching
- Somehow use the zones/fields? IDK how tho. But i feel phrase searching will yield better results than using zones/fields.