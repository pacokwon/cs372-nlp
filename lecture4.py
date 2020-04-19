import nltk
from nltk.corpus import words, gutenberg, cmudict


def unusual_words(text):
    text_vocab = {w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha()}
    english_vocab = {w.lower() for w in words.words() if w.isalpha()}

    unusual = text_vocab - english_vocab

    return sorted(unusual)


def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words("english")
    content = [w for w in text if w not in stopwords]

    return len(content) / len(text)


def solve_puzzle(letters, obligatory):
    puzzle_letters = nltk.FreqDist(letters)
    return [
        w
        for w in words.words()
        if len(w) >= 4 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters
    ]


def stress_pattern(pron):
    return [char for phone in pron for char in phone if char.isdigit()]


if __name__ == "__main__":
    # entries = cmudict.entries()
    # for word, pron in entries:
    #     if len(pron) == 3:
    #         ph1, ph2, ph3 = pron
    #         if ph1 == "P" and ph3 == "T":
    #             print(word, ph2, end=" ")

    entries = cmudict.entries()
    p3 = [
        (f"{pron[0]}-{pron[2]}", word)
        for word, pron in entries
        if pron[0] == "P" and len(pron) == 3
    ]

    cfd = nltk.ConditionalFreqDist(p3)
    for template in sorted(cfd.conditions()):
        if len(cfd[template]) > 10:
            words = sorted(cfd[template])
            wordstring = " ".join(words)
            print(template, wordstring[:70] + "...")
