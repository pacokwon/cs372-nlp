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


def parse_synset_name(synset):
    *word, pos, num = synset.name().split(".")
    return {"pos": pos, "word": ".".join(word), "id": num}


def extract_pos(synsets):
    return {parse_synset_name(synset)["pos"] for synset in synsets}


def exact_synsets(word):
    return [
        synset
        for synset in wn.synsets(word)
        if parse_synset_name(synset)["word"] == word
    ]


def heteronym_candidates():
    entries = nltk.corpus.cmudict.entries()  # -> (meaning, list[pronunciations])

    prev_word = ""
    het_cand = {}  # -> (meaning, list[list[pronunciations]])
    for idx, (word, pron_list) in enumerate(entries):
        if prev_word == word:
            if word in het_cand:
                het_cand[word].append(pron_list)
            else:
                het_cand[word] = [entries[idx - 1][1], entries[idx][1]]
        prev_word = word

    return het_cand


def phonym_difference(word, pron1, pron2):
    longer, shorter = (pron1, pron2) if len(pron1) > len(pron2) else (pron2, pron1)
    diff = []
    if len(longer) == len(shorter):
        lidx, sidx = 0, 0

        while lidx < len(longer):
            if longer[lidx] != shorter[sidx] and not (
                longer[lidx][-1].isdigit()
                and shorter[sidx][-1].isdigit()
                and longer[lidx][:-1] == shorter[sidx][:-1]
            ):
                diff.append((longer[lidx], shorter[sidx]))
            lidx += 1
            sidx += 1
    else:
        lidx, sidx = 0, 0
        while sidx < len(shorter):
            try:
                fidx = longer[lidx:].index(shorter[sidx])
            except ValueError:
                fidx = -1
            if fidx == -1 and shorter[sidx][-1].isdigit():
                try:
                    fidx = longer[lidx:].index(shorter[sidx][:-1])
                except ValueError:
                    fidx = -1

            if fidx == -1:
                diff.append(shorter[sidx])
            else:
                diff += longer[lidx : lidx + fidx]
                lidx += fidx + 1

            sidx += 1

        while lidx < len(longer):
            diff.append(longer[lidx])
            lidx += 1

    return diff


def wo_stress(phonym):
    if phonym[-1].isdigit():
        return phonym[:-1]
    else:
        return phonym


def vowel_blacklist(tup):
    blacklist = [("EH", "IH"), ("AH", "IH"), ("AA", "AO"), ("NG", "N")]

    for el in blacklist:
        if (
            tup == el
            or (wo_stress(tup[0]), wo_stress(tup[1])) == el
            or (tup[1], tup[0]) == el
            or (wo_stress(tup[1]), wo_stress(tup[0])) == el
        ):
            return True
    return False


def filter_single_def(het_cand, names, stopwords):
    filtered_het_cand = {}
    print(len(het_cand))
    keys = list(het_cand.keys())
    cnt = 0
    for key in keys:
        if key in stopwords:
            continue

        _exact_synsets = exact_synsets(key)
        pos_set = extract_pos(_exact_synsets)

        print(f"{cnt}:", end=" ")
        if len(parse_wiktionary(key)) > 1:
            filtered_het_cand[key] = {
                "pronunciations": het_cand[key],
                "pos": list(pos_set),
            }

        cnt += 1

    return filtered_het_cand


def parse_wiktionary(word):
    filename = f"./html/{word}.html"

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

    print(word)
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
                    print("hello")
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


def parse_coca(filename, het_cand):
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
                hets = [token for token in tokenized if token in het_cand]
                if hets:
                    sentences.append((sent, hets, len(hets)))
            tmp = []
        else:
            tmp.append(re.sub(re_strip_tags, "", content.lower().strip()))
        cnt += 1
    print("")
    file.close()

    return sentences


def test(het_dict):
    with open("heteronyms.json") as f:
        heteronyms = json.load(f)

    cnt = 0
    for heteronym in heteronyms:
        if heteronym in het_dict:
            cnt += 1
        print(f"{heteronym}: {heteronym in het_dict}")

    return cnt


def improvements(before, after):
    with open("heteronyms.json") as f:
        heteronyms = json.load(f)

    cnt = 0
    for heteronym in heteronyms:
        if heteronym in before and heteronym not in after:
            cnt += 1
            print(heteronym)

    return cnt


def crawl_subreddit(subreddit, heteronyms_list):
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
        hets = [token for token in tokenized if token in heteronyms_list]
        if hets:
            sents.append((tokenized, hets, len(hets)))

        for sent in nltk.sent_tokenize(re.sub(r"\n{1,}", ". ", submission.selftext)):
            tokenized = [
                re.sub(re_strip_tags, "", word.lower().strip())
                for word in nltk.word_tokenize(sent)
            ]
            hets = [token for token in tokenized if token in heteronyms_list]
            if hets:
                sents.append((tokenized, hets, len(hets)))

        count += 1
    print("")

    return sents


def crawl_coca(corpus, heteronyms_list):
    return parse_coca(corpus, heteronyms_list)


def extract_sentences(corpus_type, corpus_name, heteronyms_list):
    path = "./crawled"
    if not os.path.exists(path):
        os.mkdir(path)
    filename = f"{path}/{corpus_type}-{corpus_name}.json"

    if os.path.exists(filename):
        with open(filename, "r") as fp:
            crawled = json.load(fp)
    else:
        if corpus_type == CorpusType.COCA:
            crawled = crawl_coca(corpus_name, heteronyms_list)
        else:
            crawled = crawl_subreddit(corpus_name, heteronyms_list)

        with open(filename, "w") as fp:
            json.dump(crawled, fp, ensure_ascii=False, indent=4)

    return crawled


def max_score(sents):
    return max(sent[2] for sent in sents)


def limit(sents, num):
    return [sent for sent in sents if sent[2] >= num]


def retrieve_pos(tagger, sentence, heteronyms):
    tagged = tagger.tag(sentence)
    tagged_idx = 0

    def find_tuple(tlist, first):
        for idx, (word, _) in enumerate(tlist):
            if word == first:
                return idx

        return -1

    pos_list = []
    for heteronym in heteronyms:
        fidx = find_tuple(tagged[tagged_idx:], heteronym)
        pos_list.append((heteronym, tagged[tagged_idx + fidx][1]))
        tagged_idx += fidx + 1

    return pos_list


def num_of_pos(pos_list):
    pos_dict = {}
    for word, pos in pos_list:
        if word not in pos_dict:
            pos_dict[word] = {pos}
        else:
            pos_dict[word].add(pos)

    return sum(len(pos_dict[key]) for key in pos_dict)


def stress_sequence(phlist):
    return [int(ph[-1]) for ph in phlist if ph[-1].isdigit()]


def syllables(heteronyms, word):
    pronunciations = heteronyms[word]["pronunciations"]
    pron_dict = defaultdict(int)
    for pron in pronunciations:
        pron_dict[len(stress_sequence(pron))] += 1

    max_val = max(pron_dict.values())
    max_keys = [k for k, v in pron_dict.items() if v == max_val]
    return max_keys[0]


def judge_pronunciation(heteronyms, word, pos):
    pronunciations = heteronyms[word]["pronunciations"]
    syllables_ = syllables(heteronyms, word)

    if syllables_ == 1 or syllables_ == 2:
        sorted_pronunciations = sorted(
            pronunciations, key=lambda x: stress_sequence(x), reverse=True
        )
        if POSCategorizer.is_noun(pos):
            return sorted_pronunciations[0]

        elif POSCategorizer.is_verb(pos):
            return sorted_pronunciations[-1]

    elif syllables_ == 3:
        if POSCategorizer.is_noun(pos) or POSCategorizer.is_adjective(pos):
            return sorted(
                pronunciations, key=lambda x: stress_sequence(x)[0], reverse=True
            )[0]
        elif POSCategorizer.is_verb(pos):

            def key_func(x):
                stress = stress_sequence(x)
                return (stress[0], stress[2], stress[1])

            return sorted(pronunciations, key=key_func, reverse=True)[0]

    else:
        return pronunciations[0]


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
"""
wlp_fic
wlp_tvm
wlp_blog
wlp_news
WordAvalanches
wordplay
humor
Showerthoughts
oneliners
whowouldwin
"""

if os.path.exists("./heteronyms_list.json"):
    with open("./heteronyms_list.json", "r") as fp:
        heteronyms_list = json.load(fp)
else:
    names = [name.lower() for name in nltk.corpus.names.words()]
    stopwords = [word.lower() for word in stopwords.words()]
    heteronyms_list = filter_single_def(heteronym_candidates(), names, stopwords)
    with open("./heteronyms_list.json", "w") as fp:
        json.dump(heteronyms_list, fp, ensure_ascii=False, indent=4)

sents = []
for corpus_type, corpus_name in corpora:
    sents += extract_sentences(corpus_type, corpus_name, heteronyms_list)

tagger = get_tagger()
tagged_sents = [
    (sent[0], retrieve_pos(tagger, sent[0], sent[1]), sent[2]) for sent in sents
]
sorted_sents = sorted(
    tagged_sents, key=lambda x: (x[2], num_of_pos(x[1])), reverse=True
)
final = [
    (
        " ".join(sent[0]),
        list(
            map(
                lambda x: (*x, judge_pronunciation(heteronyms_list, x[0], x[1])),
                sent[1],
            )
        ),
        sent[2],
    )
    for sent in sorted_sents
]
