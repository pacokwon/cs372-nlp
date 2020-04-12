from nltk import FreqDist, bigrams
from nltk.book import text1, text2

if __name__ == "__main__":
    # fdist1 = FreqDist(text1)
    # V = set(text1)
    # long_words = {w for w in V if len(w) > 12 and fdist1[w] > 7}
    # print(long_words)
    print("; ".join(text1.collocation_list()))
    print("; ".join(text2.collocation_list()))
