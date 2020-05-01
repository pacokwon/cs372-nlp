#!/usr/bin/env python
# coding: utf-8

import nltk
from pprint import pprint
from nltk.corpus import brown, masc_tagged, wordnet, gutenberg, reuters
from nltk.corpus import stopwords
from functools import reduce
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle


class POSCategorizer:
    @staticmethod
    def is_adverb(pos):
        return pos in ["RB", "RBR", "RBT"]

    @staticmethod
    def is_adjective(pos):
        return pos.startswith("JJ")

    @staticmethod
    def is_verb(pos):
        return pos.startswith("VB")

    @staticmethod
    def is_noun(pos):
        return pos.startswith("NN")

    @classmethod
    def is_qualified_pos_pair(cls, word1, word2):
        pos1 = word1[1]
        pos2 = word2[1]

        pos_pair = [
            (cls.is_adverb, cls.is_adjective),
            # (cls.is_adverb, cls.is_adverb),
            # (cls.is_adverb, cls.is_verb),
            # (cls.is_adjective, cls.is_noun),
        ]

        return any([func1(pos1) and func2(pos2) for func1, func2 in pos_pair])


def filtered_bigrams(tagged_words, tagger):
    """
    Returns a list of bigrams that are qualified POS pairs

    :param tagged_words: a list of words that are (token, pos) tuples
    :type tagged_words: a list of tuples
    :returns: a list of bigrams that are (token1, token2) tuples
    :rtype: a list of tuples
    """
    bigrams = nltk.bigrams(tagged_words)

    return [
        (bigram[0][0], bigram[1][0])
        for bigram in bigrams
        if POSCategorizer.is_qualified_pos_pair(*bigram)
        and tagger.tag((bigram[0][0], bigram[1][0])) == list(bigram)
    ]


def get_tagged_words():
    """
    Returns all alphabetical words from a given corpus except stopwords

    :returns: a list of alphabetical words that are not stopwords
    :rtype: a list
    """
    english_stopwords = stopwords.words("english")
    return (
        (word[0].lower(), word[1])
        for word in brown.tagged_words() + masc_tagged.tagged_words() + tagger.tag
        if word[0].isalpha() and word[0] not in english_stopwords
    )


def generate_scores(cfd, limit):
    sid = SentimentIntensityAnalyzer()
    return {
        (condition, modified): {
            "score": cfd[condition][modified]
            / len(cfd[condition])
            * abs(
                sid.polarity_scores(f"{condition} {modified}")["compound"]
                - sid.polarity_scores(modified)["compound"]
            ),
            "frac": f"{cfd[condition][modified]} / {len(cfd[condition])}",
            "sentiment_before": sid.polarity_scores(modified),
            "sentiment_after": sid.polarity_scores(f"{condition} {modified}"),
            "increased": sid.polarity_scores(f"{condition} {modified}")["compound"]
            > sid.polarity_scores(modified)["compound"],
        }
        for condition in cfd.keys()
        for modified in cfd[condition]
        if len(cfd[condition]) < limit
    }


def get_tagger():
    try:
        with open("tagger.pkl", "rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError):
        sents = brown.tagged_sents()
        t0 = nltk.DefaultTagger("NN")
        t1 = nltk.UnigramTagger(sents, backoff=t0)
        t2 = nltk.BigramTagger(sents, backoff=t1)
        with open("tagger.pkl", "wb") as file:
            pickle.dump(t2, file, -1)
        return t2


def add_corpus_entries(cfd, *corpora):
    """
    Add words of given corpora to update the given ConditionalFreqDist object

    :param cfd: Pre constructed ConditionalFreqDist object
    :type cfd: ConditionalFreqDist
    :param *corpora: nltk.corpus objects
    :type *corpora: a list of nltk.corpus objects
    """
    words = reduce(lambda x, y: x + y, corpora, [])
    for bigram in nltk.bigrams(words):
        if bigram[0] in cfd.keys() and bigram[1] in cfd[bigram[0]]:
            # print(bigram, cfd[bigram[0]][bigram[1]])
            cfd[bigram[0]][bigram[1]] += 1
            # print(bigram, cfd[bigram[0]][bigram[1]])


if __name__ == "__main__":
    tagger = get_tagger()
    tagged_words = get_tagged_words()
    bigrams = filtered_bigrams(tagged_words, tagger)

    cfd = nltk.ConditionalFreqDist(bigrams)
    # add_corpus_entries(cfd, gutenberg, reuters)

    score_pairs = generate_scores(cfd, 30)
    filtered = {
        key: score_pairs[key] for key in score_pairs if score_pairs[key]["increased"]
    }
    sorted_pairs = [
        (k, v["score"])
        for k, v in sorted(filtered.items(), key=lambda x: x[1]["score"], reverse=True)
    ]

    for idx, pair in enumerate(sorted_pairs):
        print(idx, pair)
