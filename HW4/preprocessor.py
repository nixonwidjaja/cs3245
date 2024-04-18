from typing import Iterator

import nltk


class Preprocessor:
    """Handles the preprocessing of text/documents."""

    stemmer = nltk.PorterStemmer()

    @staticmethod
    def to_token_stream(text: str) -> Iterator[str]:
        """Tokenize a text, applying the below preprocessing before yielding
        the tokens one-by-one.

        Preprocessing applied in order of execution:
        - Sentence tokenization using `nltk.sent_tokenize`.
        - Word tokenization using `nltk.word_tokenize`.
        - Stemming using `nltk.PorterStemmer`.
        - Case-folding to lowercase.
        """
        for sentence in nltk.sent_tokenize(text):
            for token in nltk.word_tokenize(sentence):
                yield Preprocessor.stemmer.stem(token).lower()

    @staticmethod
    def file_to_token_stream(filepath: str) -> Iterator[str]:
        """Read the ENTIRE file at `filepath`, preprocessing using
        `Preprocessor.to_token_stream` to yield the tokens one-by-one.

        Args:
            filepath (str): Path to the file to tokenize.

        Yields:
            Iterator[str]: Tokens from the file.
        """
        with open(filepath, "r") as file:
            doc_text = file.read()
            yield from Preprocessor.to_token_stream(doc_text)
