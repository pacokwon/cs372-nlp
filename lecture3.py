from nltk.corpus import brown, reuters, inaugural
from nltk.corpus import PlaintextCorpusReader
from nltk import ConditionalFreqDist

if __name__ == "__main__":
    # cfd = ConditionalFreqDist(
    #     (genre, word)
    #     for genre in brown.categories()
    #     for word in brown.words(categories=genre)
    # )

    # genres = ["news", "religion", "hobbies", "science_fiction", "romance", "humor"]
    # modals = ["can", "could", "may", "might", "must", "will"]

    # cfd.tabulate(conditions=genres, samples=modals)

    # cfd = ConditionalFreqDist(
    #     (target, fileid[:4])
    #     for fileid in inaugural.fileids()
    #     for word in inaugural.words(fileid)
    #     for target in ["citizen", "america"]
    #     if word.lower().startswith(target)
    # )

    # cfd.plot()

    corpus_root = "/Users/pacokwon/course/nlp"
    wordlists = PlaintextCorpusReader(corpus_root, ".*")
    print(wordlists.words("in"))
