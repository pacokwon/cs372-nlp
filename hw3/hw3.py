from bs4 import BeautifulSoup
import nltk
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import os.path
import json
import requests
import re


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

    # if tup == ("AO1", "AA1"):
    #     print(wo_stress(tup[0]), wo_stress(tup[1]))
    #     print(wo_stress(tup[1]), wo_stress(tup[0]))
    #     print(blacklist[2])

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
        pos_set = list(extract_pos(_exact_synsets))

        # if len(het_cand[key]) == 2:  # exactly two pronunciations
        #     phdiff = phonym_difference(key, het_cand[key][0], het_cand[key][1])
        #     # 1072 with second condition, 226 without
        #     if (
        #         len(phdiff) > 1
        #         and len(parse_wiktionary(key)) > 1
        #         or (
        #             len(phdiff) == 1
        #             and type(phdiff[0]) is tuple
        #             and not vowel_blacklist(phdiff[0])
        #         )
        #     ):
        #         filtered_het_cand[key] = {
        #             "pronunciations": het_cand[key],
        #             "pos": pos_set,
        #         }
        # else:
        #     filtered_het_cand[key] = {
        #         "pronunciations": het_cand[key],
        #         "pos": pos_set,
        #     }

        print(f"{cnt}:", end=" ")
        if len(parse_wiktionary(key)) > 1:
            filtered_het_cand[key] = {
                "pronunciations": het_cand[key],
                "pos": pos_set,
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

                    ipa_all = anchor.find_parent("li").find_all("span", class_="IPA")
                    ipas |= {ipa.text for ipa in ipa_all}
                    tag = tag.next_sibling.next_sibling

            else:
                anchor = tag.next_sibling.find(href=re_us)
                if anchor == None:
                    anchor = tag.next_sibling.find(href=re_fallback)
                if anchor == None:
                    tag = tag.next_sibling
                    continue
                ipa_all = anchor.find_parent("li").find_all("span", class_="IPA")
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
                        ipa_all = anchor.find_parent("li").find_all(
                            "span", class_="IPA"
                        )
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

                    ipa_all = anchor.find_parent("li").find_all("span", class_="IPA")
                    ipas |= {ipa.text for ipa in ipa_all}
                    tag = tag.next_sibling.next_sibling

        tag = tag.next_sibling

    return list(ipas)


def parse_coca(filename, het_cand):
    file = open(filename, "r", encoding="ISO-8859-1")

    sentences = []
    tmp = []
    for line in file:
        if len(sentences) > 100:
            break

        content = line.split()[1]
        if "@" in content:
            continue
        if content == "-":
            for sent in nltk.sent_tokenize(" ".join(tmp)):
                hets = []
                for token in nltk.word_tokenize(sent):
                    if token in het_cand:
                        hets.append(token)
                if hets:
                    # print(hets)
                    sentences.append((sent, hets, len(hets)))
            tmp = []
        else:
            tmp.append(content)

    file.close()

    return sentences


def find(fword):
    entries = cmudict.entries()
    s = []
    for idx, (word, pron) in enumerate(entries):
        if word == fword:
            s.append(pron)
    return s


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


if os.path.exists("./sents.json"):
    with open("./sents.json", "r") as fp1, open("./heteronyms_list.json", "r") as fp2:
        heteronyms_list = json.load(fp2)
        sents = json.load(fp1)
else:
    if os.path.exists("./heteronyms_list.json"):
        with open("./heteronyms_list.json", "r") as fp:
            heteronyms_list = json.load(fp)
    else:
        names = [name.lower() for name in nltk.corpus.names.words()]
        stopwords = [word.lower() for word in stopwords.words()]
        het_cand = heteronym_candidates()
        heteronyms_list = filter_single_def(het_cand, names, stopwords)
        with open("./heteronyms_list.json", "w") as fp:
            json.dump(heteronyms_list, fp, ensure_ascii=False, indent=4)

    keys = list(heteronyms_list.keys())

    sents = parse_coca("wlp_tvm.txt", heteronyms_list)
    with open("sents.json", "w") as fp:
        json.dump(sents, fp, ensure_ascii=False, indent=4)

# cnt = 0
# for heteronym in heteronyms:
#     if heteronym in het_cand:
#         cnt += 1
#     print(f"{heteronym}: {heteronym in het_cand}")
