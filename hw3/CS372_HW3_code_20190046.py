from bs4 import BeautifulSoup
from collections import defaultdict
import json
import nltk
from nltk.corpus import cmudict, brown, masc_tagged, stopwords, wordnet as wn
import os.path
import pickle
from pprint import pprint
import praw
import requests
import re


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


class CorpusType:
    """
    CorpusType class constructed to simulate the behavior of an enum
    """

    COCA = "coca"
    REDDIT = "reddit"


def heteronym_candidates():
    """
    Collect and return the first possible candidates of heteronyms

    If there are two or more pronunciations of a word in cmudict,
    store the word as a possible candidate

    :returns: candidates of heteronyms
    :rtype: str-list relationship dict
    """
    entries = nltk.corpus.cmudict.entries()

    prev_word = ""
    het_cand = {}
    for idx, (word, pron_list) in enumerate(entries):
        if prev_word == word:
            if word in het_cand:
                het_cand[word].append(pron_list)
            else:
                het_cand[word] = [entries[idx - 1][1], entries[idx][1]]
        prev_word = word

    return het_cand


def filter_heteronyms(het_cand):
    """
    Filter the candidates of heteronyms by applying various criteria

    First, words are checked to see if they are not stopwords.
    Then, they're passed to the wiktionary parser to determine if
    they have two or more pronunciations or not

    :param het_cand: current candidates of heteronyms to be filtered
    :type het_cand: str-list relationship dict
    :returns: filtered candidates of heteronyms
    :rtype: str-list relationship dict
    """

    # collect stopwords in advance to filter them
    stop_words = [word.lower() for word in stopwords.words()]

    filtered_het_cand = {}
    print("Collecting heteronym candidates...")
    print(len(het_cand))
    cnt = 0
    for key in het_cand:
        if key in stop_words:
            continue

        print(f"{cnt:>4}: {key:<20}", end="\r")
        if len(parse_wiktionary(key)) > 1:
            filtered_het_cand[key] = het_cand[key]

        cnt += 1

    return filtered_het_cand


def parse_wiktionary(word):
    """
    Wiktionary parser. Returns a list of IPA symbols for the specified word

    Contains very specific, messy html parsing code because Wiktionary
    entries are very unstructured and not general; probably written by hand
    and not auto generated from generic structured data

    :param word: word in Wiktionary to parse
    :type word: str
    :returns: a list of IPA symbols for the specified word
    :rtype: list of strings
    """
    path = "./html"
    if not os.path.exists(path):
        os.mkdir(path)

    filename = f"{path}/{word}.html"
    # construct soup
    if os.path.exists(filename):
        with open(filename) as fp:
            soup = BeautifulSoup(fp, "lxml")
    else:
        html_content = requests.get(
            f"https://en.wiktionary.org/wiki/{word}"
        ).text.replace("\n", "")
        with open(filename, "w") as fp:
            fp.write(html_content)
        soup = BeautifulSoup(html_content, "lxml")

    ipas = set()
    eng_tag = soup.find("span", id="English")
    if eng_tag == None:
        return []
    tag = eng_tag.parent.next_sibling

    re_us = re.compile("(?:General_American|American_English)")
    re_fallback = re.compile("(?:English_pronunciation)")

    while tag and tag.name != "h2":
        if tag.name == "h4" and "Pronunciation" in tag.text:
            if tag.next_sibling.name == "dl":
                tag = tag.next_sibling
                while (
                    tag.name == "dl"
                    and any(["Noun" in tag.text, "Verb" in tag.text])
                    and tag.next_sibling.name == "ul"
                ):
                    anchor = tag.next_sibling.find(href=re_us)
                    if anchor == None:
                        anchor = tag.next_sibling.find(href=re_fallback)
                    if anchor == None:
                        tag = tag.next_sibling
                        continue

                    ipa_all = [anchor.find_parent("li").find("span", class_="IPA")]
                    ipas |= {ipa.text for ipa in ipa_all}
                    tag = tag.next_sibling.next_sibling

            else:
                anchor = tag.next_sibling.find(href=re_us)
                if anchor == None:
                    anchor = tag.next_sibling.find(href=re_fallback)
                if anchor == None:
                    tag = tag.next_sibling
                    continue
                ipa_all = [anchor.find_parent("li").find("span", class_="IPA")]
                ipas |= {ipa.text for ipa in ipa_all}

        elif tag.name == "h3" and "Pronunciation" in tag.text:
            if tag.next_sibling.name == "div":
                tag = tag.next_sibling

            if tag.next_sibling.name == "ul":
                tag = tag.next_sibling
                for child in tag.children:
                    if any(
                        [
                            "Noun" in child.text,
                            "Verb" in child.text,
                            "noun" in child.text,
                            "verb" in child.text,
                        ]
                    ):
                        anchor = child.find(href=re_us)
                        if anchor == None:
                            anchor = child.find(href=re_fallback)
                        if anchor == None:
                            tag = tag.next_sibling
                            continue
                        ipa_all = [anchor.find_parent("li").find("span", class_="IPA")]
                        ipas |= {ipa.text for ipa in ipa_all}

            elif tag.next_sibling.name == "dl" or tag.next_sibling.name == "p":
                tag = tag.next_sibling
                while (
                    (tag.name == "dl" or tag.name == "p")
                    and any(
                        [
                            "Noun" in tag.text,
                            "Verb" in tag.text,
                            "noun" in tag.text,
                            "verb" in tag.text,
                        ]
                    )
                    and tag.next_sibling.name == "ul"
                ):
                    anchor = tag.next_sibling.find(href=re_us)
                    if anchor == None:
                        anchor = tag.next_sibling.find(href=re_fallback)
                    if anchor == None:
                        tag = tag.next_sibling
                        continue

                    ipa_all = [anchor.find_parent("li").find("span", class_="IPA")]
                    ipas |= {ipa.text for ipa in ipa_all}
                    tag = tag.next_sibling.next_sibling

        tag = tag.next_sibling

    return list(ipas)


def parse_coca(filename, heteronyms):
    """
    Returns tokenized sentences of the specified file from the COCA corpus.
    The sentences contain at least one heteronym, heteronyms being defined
    by the content of param heteronyms

    :param filename: name of text file. ex> wlp_blog
    :type filename: str
    :param heteronyms: user-collected heteronyms information
    :type heteronyms: str - list relationship dict
    :returns: sentences, containing each word as an element
    :rtype: list of lists of strings
    """
    file = open(f"./corpora/{filename}.txt", "r", encoding="ISO-8859-1")
    sentences = []
    re_strip_tags = re.compile("<.*?>")

    tmp = []
    cnt = 0
    for line in file:
        print(f"Sentence no. {cnt}", end="\r")
        try:
            content = line.split()[1]
        except IndexError:
            continue
        if "@" in content:
            continue
        if content == "-":
            for sent in nltk.sent_tokenize(" ".join(tmp)):
                tokenized = nltk.word_tokenize(sent)
                hets = [token for token in tokenized if token in heteronyms]
                if hets:
                    sentences.append(
                        (tokenized, hets, len(hets), f"coca:{filename}:line {cnt}")
                    )
            tmp = []
        else:
            tmp.append(re.sub(re_strip_tags, "", content.lower().strip()))
        cnt += 1
    print("")
    file.close()

    return sentences


def parse_reddit(subreddit, heteronyms):
    """
    Returns tokenized sentences of the latest posts' titles and
    contents from the specified subreddit. The sentences contain
    at least one heteronym, heteronyms being defined by the content
    of param heteronyms

    :param subreddit: name of subreddit
    :type subreddit: str
    :param heteronyms: user-collected heteronyms information
    :type heteronyms: str - dict relationship dict
    :returns: sentences, containing each word as an element
    :rtype: list of lists of strings
    """
    reddit = praw.Reddit(
        client_id="4SQSJJQWux5P4A",
        client_secret="ytoF1luquMEAj5DIECftSl1KvGI",
        user_agent="cs372app",
        username="pacokwon",
        password="enseitankado",
    )

    re_strip_tags = re.compile("<.*?>")
    sents = []
    count = 0
    for submission in reddit.subreddit(subreddit).new(limit=None):
        print(f"Sentence no. {count}", end="\r")
        tokenized = [
            re.sub(re_strip_tags, "", word.lower().strip())
            for word in nltk.word_tokenize(submission.title)
        ]
        hets = [token for token in tokenized if token in heteronyms]
        if hets:
            sents.append(
                (tokenized, hets, len(hets), f"reddit:r/{subreddit}:{submission.url}")
            )

        for sent in nltk.sent_tokenize(re.sub(r"\n{1,}", ". ", submission.selftext)):
            tokenized = [
                re.sub(re_strip_tags, "", word.lower().strip())
                for word in nltk.word_tokenize(sent)
            ]
            hets = [token for token in tokenized if token in heteronyms]
            if hets:
                sents.append(
                    (
                        tokenized,
                        hets,
                        len(hets),
                        f"reddit:r/{subreddit}:{submission.url}",
                    )
                )

        count += 1
    print("")

    return sents


def extract_sentences(corpus_type, corpus_name, heteronyms_list):
    """
    Top level function that returns sentences that contain at least
    one heteronym from the corpus specified by arguments

    :param corpus_type: type of corpus. either reddit or coca
    :type corpus_type: str
    :param corpus_name: specific filename or subreddit name of corpus
    :type corpus_name: str
    :returns: list of sentences
    :rtype: list of lists of strings
    """
    print(f"Collecting sentences from {corpus_type}-{corpus_name}...")

    path = "./crawled"
    if not os.path.exists(path):
        os.mkdir(path)
    filename = f"{path}/{corpus_type}-{corpus_name}.json"

    if os.path.exists(filename):
        with open(filename, "r") as fp:
            crawled = json.load(fp)
    else:
        if corpus_type == CorpusType.COCA:
            crawled = parse_coca(corpus_name, heteronyms_list)
        else:
            crawled = parse_reddit(corpus_name, heteronyms_list)

        with open(filename, "w") as fp:
            json.dump(crawled, fp, ensure_ascii=False, indent=4)

    return crawled


def retrieve_pos(tagger, sentence, heteronyms):
    """
    Return a list of tuples that specify the pos of
    the respective heteronyms

    :param tagger: pos tagger
    :type tagger: nltk.BigramTagger object
    :param sentence: pos tagged sentence that contains heteronyms
    :type sentence: list of strings
    :param heteronyms: list of heteronyms to be pos tagged
    :type heteronyms: list of strings
    :returns: list of tuples that specify each heteronyms' POS
    :rtype: list of 2 element-tuples.
            first element is the heteronym, second element is the POS
    """
    tagged = tagger.tag(sentence)
    tagged_idx = 0

    def find_tuple(tlist, first):
        for idx, (word, *_) in enumerate(tlist):
            if word == first:
                return idx
        return -1

    pos_list = []
    for heteronym in heteronyms:
        fidx = find_tuple(tagged[tagged_idx:], heteronym)
        pos_list.append((heteronym, tagged[tagged_idx + fidx][1]))
        tagged_idx += fidx + 1

    return pos_list


def num_of_words(word_pos_tuples):
    """
    Get the number of distinct words from the list of tuples

    for example the argument might look something like:
    [(wind, NN), (wind, VB), (tear, NN), (tear, VB), (wind, VB)]
    then the result of the function would be 4, since there are
    4 distinct words in this list

    :param word_pos_tuples: list of tuples, where the first element
                            is a word, and the second element is its POS
    :type word_pos_tuples: list of tuples
    :returns: number of distinct words from the list of tuples
    :rtype: int
    """
    pos_dict = {}
    for word, pos in word_pos_tuples:
        if word not in pos_dict:
            pos_dict[word] = {pos}
        else:
            pos_dict[word].add(pos)

    return sum(len(pos_dict[key]) for key in pos_dict)


def stress_sequence(phlist):
    """
    Return the stress integer sequence from the list of phonemes

    For example:
    the argument ['AH0', 'B', 'R', 'IH1', 'JH', 'D']
    yields the result [0, 2]
    Note that the integer is larger in this order of stress: Primary > Secondary > None
    The original ARPABET specifies that 1 is the primary stress, but here I use 2 as
    primary stress and 1 as secondary stress

    :param phlist: list of phonemes. the same format as the one in cmudict
    :type phlist: list of strings
    :returns: stress integer sequence
    :rtype: list of integers
    """
    mapping = {"0": 0, "1": 2, "2": 1}
    return [mapping[ph[-1]] for ph in phlist if ph[-1].isdigit()]


def syllables(heteronyms, word):
    """
    Retrieve the number of syllables of a specified word

    :param heteronyms: user-collected heteronyms information
    :type heteronyms: string - list relationship dict
    :param word: the word whose number of sylllables is desired to be known
    :type word: str
    :returns: number of syllables of word
    :rtype: int
    """
    pronunciations = heteronyms[word]
    pron_dict = defaultdict(int)
    for pron in pronunciations:
        pron_dict[len(stress_sequence(pron))] += 1

    max_val = max(pron_dict.values())
    max_keys = [k for k, v in pron_dict.items() if v == max_val]
    return max_keys[0]


def judge_pronunciation(heteronyms, word, pos):
    """
    Judge and return the pronunciation of a given
    word by looking at its POS

    :param heteronyms: user-collected heteronyms information
    :type heteronyms: string - list relationship dict
    :param word: the word whose pronunciation is desired to be known
    :type word: str
    :param pos: POS of given word
    :type pos: str
    :returns: pronunciation of given word
    :rtype: list of strings
    """
    pronunciations = heteronyms[word]
    syllables_ = syllables(heteronyms, word)

    if syllables_ == 1 or syllables_ == 2:
        sorted_pronunciations = sorted(
            pronunciations, key=lambda x: stress_sequence(x), reverse=True
        )
        if POSCategorizer.is_noun(pos):
            return sorted_pronunciations[0]

        elif POSCategorizer.is_verb(pos):
            return sorted_pronunciations[-1]

        else:
            return sorted_pronunciations[0]

    elif syllables_ == 3:
        if POSCategorizer.is_noun(pos) or POSCategorizer.is_adjective(pos):
            return sorted(
                pronunciations, key=lambda x: stress_sequence(x)[0], reverse=True
            )[0]
        elif POSCategorizer.is_verb(pos):

            def key_func(x):
                stress = stress_sequence(x)

                if len(stress) == 3:
                    return (stress[0], stress[2], stress[1])
                elif len(stress) == 2:
                    return (stress[0], stress[1])

            return sorted(pronunciations, key=key_func, reverse=True)[0]

    else:
        return pronunciations[0]


def write_to_csv(filename, results):
    """
    Save result sentences into a csv format file

    :param filename: expected file name of csv file
    :type filename: str
    :param results: list of sentences to be saved
    :type results: list of tuples
    """
    with open(filename, "w") as fp:
        for sent in results:
            sentence = sent[0]
            pronunciations_list = []
            for word, pos, pron in sent[1]:
                merged = "-".join(phoneme for phoneme in pron)
                pronunciations_list.append(f"{word} / {pos} / {merged}")
            pronunciations = ",".join(pron for pron in pronunciations_list)
            source = sent[3]

            print(f'"{sentence}", {pronunciations}, {source}', file=fp)


corpora = [
    (CorpusType.REDDIT, "whowouldwin"),
    (CorpusType.REDDIT, "WordAvalanches"),
    (CorpusType.REDDIT, "wordplay"),
    (CorpusType.REDDIT, "humor"),
    (CorpusType.REDDIT, "Showerthoughts"),
    (CorpusType.REDDIT, "oneliners"),
    (CorpusType.COCA, "wlp_fic"),
    (CorpusType.COCA, "wlp_tvm"),
    (CorpusType.COCA, "wlp_blog"),
    (CorpusType.COCA, "wlp_news"),
]


def main():
    # 1. Constructing a list of heteronyms. Load from file if present
    if os.path.exists("./heteronyms_list.json"):
        with open("./heteronyms_list.json", "r") as fp:
            heteronyms = json.load(fp)
    else:
        heteronyms = filter_heteronyms(heteronym_candidates())
        with open("./heteronyms_list.json", "w") as fp:
            json.dump(heteronyms, fp, ensure_ascii=False, indent=4)

    # 2. Collecting and tagging sentences from various corpora that contain at least one heteronym
    sents = []
    for corpus_type, corpus_name in corpora:
        sents += extract_sentences(corpus_type, corpus_name, heteronyms)
    tagger = get_tagger()

    # 3. Assigning pronunciations to heteronyms
    tagged_sents = [
        (sent[0], retrieve_pos(tagger, sent[0], sent[1]), sent[2], sent[3])
        for sent in sents
    ]

    # 4. ranking the sentences
    sorted_sents = sorted(
        tagged_sents, key=lambda x: (x[2], num_of_words(x[1])), reverse=True
    )
    final = [
        (
            " ".join(sent[0]),
            list(
                map(
                    lambda x: (*x, judge_pronunciation(heteronyms, x[0], x[1])),
                    sent[1],
                )
            ),
            sent[2],
            sent[3],
        )
        for sent in sorted_sents
        if len(" ".join(sent[0])) < 200
    ]

    write_to_csv("results.csv", final[:30])


if __name__ == "__main__":
    main()
