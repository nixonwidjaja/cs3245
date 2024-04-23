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

    PREPROCESSING_MODE: Literal["stem", "lemma_wo_pos", "lemma_with_pos", "lemma_with_pos_and_position"] = "lemma_with_pos_and_position"

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
                        
            case "lemma_with_pos_and_position":
                for sentence in nltk.sent_tokenize(text):
                    for i, part in enumerate(nltk.pos_tag(nltk.word_tokenize(sentence))):
                        token, pos = part
                        wordnet_pos = convert_pos_to_wordnet_pos(pos)
                        yield (i, Preprocessor.lemmatizer.lemmatize(token, wordnet_pos).lower())


if __name__ == "__main__":
    print(Preprocessor.PREPROCESSING_MODE)
