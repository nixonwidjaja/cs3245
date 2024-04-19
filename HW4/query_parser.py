import re

import nltk

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)

from typing import cast

from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from preprocessor import Preprocessor, convert_pos_to_wordnet_pos


class QueryParser:
    @staticmethod
    def get_query_tokens(query: str) -> list[str]:
        assert Preprocessor.PREPROCESSING_MODE in ["stem", "lemma_wo_pos", "lemma_with_pos"], \
            "We forgot to update this function after updating `Preprocessor.PREPROCESSING_MODE`."  # fmt:skip

        if is_bool_query := '"' in query or "AND" in query:
            phrases: list[str] = [x.replace('"', "") for x in re.split(r"\s+AND\s+", query)]
            output_tokens: list[str] = []
            for phrase in phrases:
                output_tokens.extend(Preprocessor.to_token_stream(phrase))
            return output_tokens

        if Preprocessor.PREPROCESSING_MODE == "lemma_with_pos":
            # Include POS during lemmatization.
            token_set: set[str] = set()
            for token, pos in nltk.pos_tag(nltk.word_tokenize(query)):
                wordnet_pos = convert_pos_to_wordnet_pos(pos)
                Preprocessor.lemmatizer.lemmatize(token, wordnet_pos).lower()
                token_set.add(token)
                token_set |= QueryParser.get_synonyms(token, wordnet_pos)
            return list(token_set)

        token_set: set[str] = set()
        for token in nltk.word_tokenize(query):
            Preprocessor.lemmatizer.lemmatize(token).lower()
            token_set.add(token)
            token_set |= QueryParser.get_synonyms(token)

        if Preprocessor.PREPROCESSING_MODE == "stem":
            return [Preprocessor.stemmer.stem(lemma) for lemma in token_set]

        return list(token_set)

    @staticmethod
    def get_synonyms(token: str, wordnet_pos: str = wordnet.NOUN) -> set[str]:
        synonyms: set[str] = set()
        for syn in cast(list[Synset], wordnet.synsets(token, wordnet_pos)):
            for lemma in cast(list[str], syn.lemma_names()):
                synonyms.add(lemma.lower())
        return synonyms
