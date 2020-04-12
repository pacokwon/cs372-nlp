from nltk.book import text1


def lexical_diversity(text):
    return len(text) / len(set(text))


def percentage(count, total):
    return 100 * count / total


if __name__ == "__main__":
    text1.concordance("monstrous")  # shows every occurence of a given word with context
    # text1.similar("monstrous")  # check for similar words in context
    # text1.concordance("contemptible")  # check for similar words in context
    text1.generate()
