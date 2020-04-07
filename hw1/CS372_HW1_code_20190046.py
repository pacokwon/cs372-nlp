from nltk.corpus import wordnet as wn
from nltk.book import text1 as sample
from itertools import product


def find_synonyms(word):
    """
    return all lemmas of synsets of the given word

    params:
        word - word string

    """
    # note that only words that are one of noun, adj, satellite adj are being included
    syns = [syn for syn in wn.synsets(word) if syn.pos() in "vas"]

    # scan all words in its hypernym paths as well
    paths = [path for syn in syns for path in syn.hypernym_paths()]
    lemmas = [lemma for path in paths for syn in path for lemma in syn.lemma_names()]

    return lemmas


def qualifies_pos(word, pos):
    """
    return whether a given word is of the given pos

    for example, if 'horse' is given as word and 'vas' is given as pos,
    since horse is neither a verb nor an adjective nor a satellite adjective,
    the function will return False.

    param:
        word - string
        pos - string containing single-letter part of speech bits ex> vas, vn
    """
    return len(set(syn.pos() for syn in wn.synsets(word)).intersection(set(pos))) > 0


def intensified_pairs(text):
    """
    find and return all 'intensified pairs' provided in the text

    intensified pair simply means a pair that has adverbs of intensity
    before or after a word

    params:
        text - nltk.text.Text object or a subscriptable value containing strings as elements
    """
    # this list contains 'adverbs of intensity' that can come before or after a word
    aod_both = [
        "extremely",
        "completely",
        "totally",
        "absolutely",
        "greatly",
        "exceedingly",
        "hugely",
        "highly",
    ]

    # this list contains 'adverbs of intensity' that comes before a word
    aod_pre = ["quite", "very", "too", "really"]

    # key: value will be stored like
    # 'extol': { 'pair': 'praise highly', 'intmod': 'highly' }
    pair_map = {}
    for idx, word in enumerate(text):
        if word in aod_both:
            # if the word that comes before is a verb (or has at least one entry that is a verb)
            # notice that when a word precedes the modifier, it can only be a verb
            if qualifies_pos(text[idx - 1], "v"):
                pair_map[text[idx - 1]] = {
                    "pair": f"{text[idx - 1]} {word}",
                    "intmod": word,
                }

            # if at least one of the entry is a verb or an adj or a satellite adj
            # in this case the modified word can be one of verb, adj & satellite adj
            if qualifies_pos(text[idx + 1], "vas"):
                pair_map[text[idx + 1]] = {
                    "pair": f"{word} {text[idx + 1]}",
                    "intmod": word,
                }

        if word in aod_pre:
            # if at least one of the entry is a verb or an adj or a satellite adj
            # notice that adverbs in aod_pre modify adj's & satellite adj's, and not verbs
            if qualifies_pos(text[idx + 1], "as"):
                pair_map[text[idx + 1]] = {
                    "pair": f"{word} {text[idx + 1]}",
                    "intmod": word,
                }

    return pair_map


def syn_pairs(text):
    """
    find and return all synonymous (word, word + intensity modifying word)
    pairs from text

    params: nltk.text.Text object or a subscriptable value containing strings as elements
    """
    pair_map = intensified_pairs(text)

    # construct a set that has only the modified words in the expression
    meaning_set = set(meaning for meaning in pair_map)

    # construct a dictionary with synonym : word key value relationship
    syn_map = {syn: word for word in meaning_set for syn in find_synonyms(word)}

    # if there is a common word between the dictionary and text, include it in the set as pairs with the expression
    return set(
        (word, pair_map[syn_map[word]]["pair"])
        for word in text
        if word in syn_map and word not in pair_map[syn_map[word]]["pair"]
    )


if __name__ == "__main__":
    # retrieve pairs
    pairs = list(syn_pairs(sample))

    # print the first 50 in a formatted manner
    print("\n".join([f"{t[0]:20}{t[1]}" for t in pairs[:50]]))
    print(len(pairs))
