from typing import Iterator, Literal

import nltk

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import wordnet


def convert_pos_to_wordnet_pos(pos: str) -> str:
    """Map POS tag to WordNet's POS tags."""
    mapping = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return mapping.get(pos[0].upper(), wordnet.NOUN)


class Preprocessor:
    """Handles the preprocessing of text/documents."""

    PREPROCESSING_MODE: Literal["stem", "lemma_wo_pos", "lemma_with_pos"] = "lemma_wo_pos"

    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    @staticmethod
    def to_token_stream(text: str) -> Iterator[str]:
        """Tokenize a text, applying the below preprocessing before yielding
        the tokens one-by-one.

        Preprocessing applied in order of execution:
        - Sentence tokenization using `nltk.sent_tokenize`.
        - Word tokenization using `nltk.word_tokenize`.
        - Stemming/Lemmatization based on `Preprocessor.PREPROCESSING_MODE`.
            - `"stem"` mode uses `nltk.PorterStemmer`.
            - `"lemma_wo_pos"` mode uses `nltk.WordNetLemmatizer` with \
                the POS defaulting to noun.
            - `"lemma_with_pos"` mode uses `nltk.WordNetLemmatizer` with \
                `nltk.pos_tag`.
        - Case-folding to lowercase.
        """
        match Preprocessor.PREPROCESSING_MODE:
            case "stem":
                for sentence in nltk.sent_tokenize(text):
                    for token in nltk.word_tokenize(sentence):
                        yield Preprocessor.stemmer.stem(token).lower()

            case "lemma_wo_pos":
                for sentence in nltk.sent_tokenize(text):
                    for token in nltk.word_tokenize(sentence):
                        yield Preprocessor.lemmatizer.lemmatize(token).lower()

            case "lemma_with_pos":
                for sentence in nltk.sent_tokenize(text):
                    for token, pos in nltk.pos_tag(nltk.word_tokenize(sentence)):
                        wordnet_pos = convert_pos_to_wordnet_pos(pos)
                        yield Preprocessor.lemmatizer.lemmatize(token, wordnet_pos).lower()

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
