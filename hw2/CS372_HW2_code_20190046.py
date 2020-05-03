#!/usr/bin/env python
# coding: utf-8

import nltk
from pprint import pprint
from nltk.corpus import brown, masc_tagged, wordnet, gutenberg, reuters, inaugural
from nltk.corpus import stopwords
from functools import reduce
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle


class POSCategorizer:
    """
    A collection of methods that classifies the pos of words
    """

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
        """
        Returns whether or not the two word's POS sequence is what we define as 'qualified'

        'Qualified' is defined as a certain pattern of two pos's that this method defines
        the local variable pos_pair defines such patterns with the class's static methods

        :param word1: first word bigram
        :type word1: a tuple of (token, pos)
        :param word2: second word bigram
        :type word2: a tuple of (token, pos)
        :returns: True if the two word's POS sequence is 'qualified', False otherwise
        :rtype: boolean
        """
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
    Returns all alphabetical words from the brown and masc_tagged corpora

    :returns: a list of words
    :rtype: a list
    """
    return (
        (word[0].lower(), word[1])
        for word in brown.tagged_words() + masc_tagged.tagged_words()
    )


def generate_scores(cfd, limit):
    """
    Returns a dictionary that contains a bigram as the key, and the score of the bigram as the value

    :param cfd: ConditionalFreqDist object that contains the occurrences of bigrams
    :type cfd: ConditionalFreqDist object
    :param limit: the maximum number(exclusive) of modified words for a modifier
    :type limit: int
    """
    sid = SentimentIntensityAnalyzer()

    def polarity_score(cond, mod):
        combined = sid.polarity_scores(f"{cond} {mod}")
        single = sid.polarity_scores(mod)

        return abs(
            combined["compound"] + combined["pos"] - single["compound"] + single["pos"]
        )

    def increased(cond, mod):
        combined = sid.polarity_scores(f"{cond} {mod}")["compound"]
        single = sid.polarity_scores(mod)["compound"]
        return combined > single and combined * single >= 0

    return {
        (condition, modified): {
            "score": cfd[condition][modified]
            / len(cfd[condition])
            * polarity_score(condition, modified)
            * 100,
            "increased": increased(condition, modified),
        }
        for condition in cfd.keys()
        for modified in cfd[condition]
        if len(cfd[condition]) < limit
    }


def get_tagger():
    """
    Return a POS tagger; generate one if it doesn't exist

    :returns: nlkt.BigramTagger object trained with brown corpora
    :rtype: nltk.BigramTagger object
    """
    try:
        with open("tagger.pkl", "rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError):
        sents = brown.tagged_sents() + masc_tagged.tagged_sents()
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
    words = reduce(lambda acc, cur: acc + cur.words(), corpora, [])
    for bigram in nltk.bigrams(words):
        if bigram[0] in cfd.keys() and bigram[1] in cfd[bigram[0]]:
            cfd[bigram[0]][bigram[1]] += 1


def write_to_csv(filename, pairs):
    """
    Save pairs into a csv format file

    :param filename: expected file name of csv file
    :type filename: string
    :param pairs: list of bigrams
    :type pairs: list of tuples of strings
    """
    with open(filename, "w") as f:
        print("\n".join(f"{pair[0]},{pair[1]}" for pair in pairs), file=f)


def main():
    """
    main function of program
    """
    tagger = get_tagger()  # retrieve trained tagger
    tagged_words = get_tagged_words()  # retrieve tagged words list

    # filter bigrams of tagged words by pos
    bigrams = filtered_bigrams(tagged_words, tagger)

    # construct a ConditionalFreqDist
    cfd = nltk.ConditionalFreqDist(bigrams)

    # collect additional text info from other corpora
    add_corpus_entries(cfd, gutenberg, reuters, inaugural)

    # grade each pairs
    score_pairs = generate_scores(cfd, 9)

    filtered = {
        key: score_pairs[key] for key in score_pairs if score_pairs[key]["increased"]
    }

    sorted_pairs = [
        k
        for k, v in sorted(filtered.items(), key=lambda x: x[1]["score"], reverse=True)
    ]
    pprint(sorted_pairs[:100])
    write_to_csv("result.csv", sorted_pairs)


if __name__ == "__main__":
    main()
