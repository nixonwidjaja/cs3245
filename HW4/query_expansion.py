from typing import cast

import nltk
from preprocessor import Preprocessor, convert_pos_to_wordnet_pos

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)


from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Lemma, Synset


def expand_query(query_tokens: list[str]) -> list[str]:
    assert Preprocessor.PREPROCESSING_MODE in ["lemma_wo_pos", "lemma_with_pos"]

    expanded_query = []
    for token, pos in nltk.pos_tag(query_tokens):
        wordnet_pos = convert_pos_to_wordnet_pos(pos)
        synonyms: list[str] = []
        for syn in cast(list[Synset], wordnet.synsets(token, wordnet_pos)):
            for lemma in cast(list[Lemma], syn.lemmas()):
                synonyms.append(lemma.name().lower())
        expanded_query.append(token)
        expanded_query.extend(list(set(synonyms)))
    return expanded_query
