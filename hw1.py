from nltk.corpus import wordnet as wn
from nltk.book import text2 as sample
from itertools import product


def find_synonyms(word):
    """
    return all lemmas of synsets of the given word

    params:
        word - string word
    """
    syns = [syn for syn in wn.synsets(word) if syn.pos() in "vas"]
    paths = [path for syn in syns for path in syn.hypernym_paths()]
    lemmas = [lemma for path in paths for syn in path for lemma in syn.lemma_names()]

    return lemmas


def highest_sim(word1, word2):
    ss1 = wn.synsets(word1)
    ss2 = wn.synsets(word2)
    scores = []
    for s1, s2 in product(ss1, ss2):
        if s1.pos() != s2.pos():
            continue
        sc = s1.wup_similarity(s2)
        if sc != None:
            scores.append(sc)

    return max(scores) if scores else 0


def qualifies_pos(word, pos):
    """
    return whether a given word is of the given pos

    for example, if 'horse' is given as word and 'vas' is given as pos,
    since horse is neither a verb nor an adjective nor a satellite adjective,
    the function will return false.

    param:
        word - string
        pos - string containing single-letter part of speech bits ex> vas, vn
    """
    return len(set(syn.pos() for syn in wn.synsets(word)).intersection(set(pos)))


def intensified_pairs(text):
    """
    find and return all 'intensified pairs' provided in the text

    intensified pair simply means a pair that has adverbs of intensity
    before or after a word

    params:
        text - nltk.text.Text object
    """
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

    aod_pre = [
        "quite",
        "very",
        "too",
        "really",
        "very",
    ]

    # key: value will be stored like
    # 'extol': { 'pair': 'praise highly', 'intmod': 'highly' }
    pair_map = {}
    for idx, word in enumerate(text):
        if word in aod_both:
            if qualifies_pos(text[idx - 1], "vas"):
                pair_map[text[idx - 1]] = {
                    "pair": f"{text[idx - 1]} {word}",
                    "intmod": word,
                }

            if qualifies_pos(text[idx + 1], "vas"):
                pair_map[text[idx + 1]] = {
                    "pair": f"{word} {text[idx + 1]}",
                    "intmod": word,
                }

        if word in aod_pre:
            if qualifies_pos(text[idx + 1], "vas"):
                pair_map[text[idx + 1]] = {
                    "pair": f"{word} {text[idx + 1]}",
                    "intmod": word,
                }

    return pair_map


def syn_pairs(text):
    """
    find and return all synonymous (word, word + intensity modifying word)
    pairs from text

    params: nltk.text.Text object
    """
    pair_map = intensified_pairs(text)
    meaning_set = set(meaning for meaning in pair_map)
    syn_map = {syn: word for word in meaning_set for syn in find_synonyms(word)}

    return set(
        (word, pair_map[syn_map[word]]["pair"])
        for word in text
        if word in syn_map and word not in pair_map[syn_map[word]]["pair"]
    )


if __name__ == "__main__":
    pairs = syn_pairs(sample)
    with open("out", "w") as f:
        f.write("\n".join(sorted([f"{t[0]:20}{t[1]}" for t in pairs])))

    print(len(pairs))

