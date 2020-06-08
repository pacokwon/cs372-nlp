import csv
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import brown, masc_tagged
import pickle


def get_tagger():
    """
    Return a POS tagger; generate one if it doesn't exist

    Using a BigramTagger, set backoff to UnigramTagger, which again has DefaultTagger as backoff
    Assuming that the words that are not familiar to the tagger would mostly be medical terms,
    and that those medical terms are mostly nouns, we set the default POS to NN. So that we have
    a higher probability of getting it right.

    :returns: nlkt.BigramTagger object trained with brown corpora
    :rtype: nltk.BigramTagger
    """
    try:
        with open("tagger.pkl", "rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError):
        sents = brown.tagged_sents() + masc_tagged.tagged_sents()
        t0 = nltk.DefaultTagger("NN")
        t1 = nltk.UnigramTagger(sents, backoff=t0)
        t2 = nltk.BigramTagger(sents, backoff=t1)
        t3 = nltk.TrigramTagger(sents, backoff=t2)
        with open("tagger.pkl", "wb") as file:
            pickle.dump(t3, file, -1)
        return t3


def word_qualified(porter, action, before, word):
    """
    Returns whether or not `word` is qualified as an action

    this function is designed to be used in the `find` function.
    it primarily checks if `word` can be qualified as `action`
    the function first checks if the stem of two words are equal,
    and then checks the word in detail such as its suffix and context

    For example, if word='inhibits' and action='inhibit', the function would return True
    but if word='inhibitor' and action='inhibit', the function would return False

    :param porter: porter stemmer object
    :type porter: nltk.PorterStemmer
    :param action: the action to be matched with `word`
    :type action: str
    :param before: the word that comes before `word` in the sentence that originally contains `word`
    :type before: str
    :param word: the word to be checked for qualification
    :type word: str
    :returns: whether or not `word` is qualified as an action
    :rtype: bool
    """
    return porter.stem(action) == porter.stem(word) and not (
        word.endswith("ing")
        or word.endswith("or")
        or word.endswith("er")
        or before == "can"
        or before == "to"
    )


def find(sentence, actions):
    """
    Find the occurrence of an action in `sentence` that is in `actions`, and return
    the word along with its corresponding index

    For example the sentence `cocaine binds to α-synuclein` would be passed into
    `sentence` as [('cocaine', 'NN'), ('binds', 'VBZ'), ('to', 'TO'), ('α-synuclein', 'NN')]
    And `actions` could be something like ['activate', 'inhibit', 'bind', 'induce', 'abolish']
    The result of the function with these as arguments would be ('binds', 1)

    :param sentence: POS tagged sentence
    :type sentence: a list of tuples, where the first element is a word and the second its POS tag
    :param actions: a list of actions to look for
    :type actions: a list of strings, where the element is one action
    :returns: the action and its corresponding index in the sentence
    :rtype: two-element tuple where first element is the action and the second its index in the sentence
    """
    porter = PorterStemmer()
    for idx, (word, _) in enumerate(sentence):
        for action in actions:
            if word_qualified(porter, action, sentence[idx - 1][0], word):
                return word, idx
    return "", -1


def tree_contents(tree):
    """
    Concatenate and return the contents of `tree`'s leaves in
    a string delimited by a single space

    For example, if the tree looks something like this:
    (CHUNK (NP (NG cancer/NN stem/NN cells/NNS)))
    the contents of its leaves are 'cancer', 'stem', 'cells'
    so the function would return them in a single string like 'cancer stem cells'

    :param tree: the tree whose leaves' contents are to be read
    :type tree: nltk.Tree
    :returns: the contents of `tree`'s leaves in a string delimited by a single space
    :rtype: str
    """
    return " ".join(leaf[0] for leaf in tree.flatten().leaves())


def extract_chunks(sent):
    """
    Extract desired 'chunks' from a given list of tagged words

    The main role of this function is to extract *Noun Phrases* from `sent`
    So we construct a grammar that retrieves Noun Phrases by the label 'CHUNK' from a sentence
    The grammar is explained in detail in the report.
    After parsing the sentence, we extract the contents of all subtrees of the label CHUNK
    and return the list of contents as a result

    :param sent: list of tagged words. A portion of a tagged sentence will be passed as argument
    :type sent: list of tuples where the first element is a word, and the second its POS tag
    :returns: list of strings that are labeled 'CHUNK' as a result of parsing `sent`
    :rtype: list of str
    """
    grammar = """
        NG: {<AT|DT>?<NP.*|NN.*|VBG>+}
        NGC: {(<NG><CC>)*<NG>}
        CHUNK:  {<IN|TO>?<NGC>}
                }<IN|TO><NGC>{
    """
    cp = nltk.RegexpParser(grammar)
    parsed = cp.parse(sent)
    chunks = []
    for subtree in parsed.subtrees():
        if subtree.label() == "CHUNK":
            chunks.append(tree_contents(subtree))
    return chunks


def get_test_sentences(filename):
    """
    Retrieve and return test sentences

    the file corresponding to the filename must be a csv file that has exactly 100 rows
    which has the following 8 fields in order:
    sentence, triplet(1), triplet(2), triplet(3), PMID, year, journal title, organization

    the triplets are manually annotated triplets
    the last 20 rows will be used as testing sentences

    :param filename: path to the csv file
    :type filename: str
    :returns: 20 csv rows containing test data
    :rtype: list of lists, each element containing each field of the csv file
    """
    with open(filename, "r") as file:
        # order: sent(0) - triplet(1, 2, 3) - cites(4, 5, 6, 7)
        reader = csv.reader(file, delimiter=",", quotechar='"')
        total = [row for row in reader]
        return total[80:]


def save_output(filename, rows, guesses):
    """
    Save the output of program to csv file

    the output csv file contains the following 11 fields in order:
    sentence, triplet(1), triplet(2), triplet(3), guess(1), guess(2), guess(3), PMID, year, journal title, organization

    the triplets are manually annotated triplets, ie the answer triplet
    the guesses are predictions made by the program

    :param filename: path to the outpuut csv file
    :type filename: str
    :param rows: rows used for testing
    :type rows: list of lists, each element containing each field from the input csv file
    :param guesses: list of guesses made for each row in `rows`
    :type guesses: list of lists, each element containing predicted triplets
    """
    with open(filename, "w") as file:
        writer = csv.writer(file, delimiter=",", quotechar='"')
        for row, guess in zip(rows, guesses):
            writer.writerow(row[:4] + guess + row[4:])


if __name__ == "__main__":
    # retrieve 20 test sentences
    test_sentences = get_test_sentences("./rand_input.csv")
    # get trained nltk.BigramTagger
    tagger = get_tagger()
    # get porter Stemmer
    porter = PorterStemmer()

    # predefined list of actions
    actions = ["activate", "inhibit", "bind", "induce", "abolish"]

    # actions that could require prepositions after them
    prep_actions = ["bind"]
    # stems of actions that could require prepositions after them
    prep_actions_stems = [porter.stem(word) for word in prep_actions]
    # candidates of prepositions, according to training data
    prep_candidates = ["to", "with"]

    # number of correct guesses
    correct_count = 0
    # list of guesses
    guesses = []

    # prediction for each test sentence
    for row in test_sentences:
        # sentence is in index 0, the triplets are in index 1 ~ 3
        sent, ans = row[0], tuple(row[1:4])

        # tag the sentence using bigram tagger
        tagged = tagger.tag(nltk.word_tokenize(sent))

        # retrieve the action word and the index of it
        action, start = find(tagged, actions)
        end = start + 1

        # if there exists preposition after the action, increment end
        if (
            porter.stem(action) in prep_actions_stems
            and tagged[end][0] in prep_candidates
        ):
            action = f"{action} {tagged[end][0]}"
            end += 1

        # list of Noun Phrases pre- action word
        pre = extract_chunks(tagged[:start])

        # list of Noun Phrases post- action word
        post = extract_chunks(tagged[end:])

        # extract the noun phrases closest to action
        guess = (pre[-1] if pre else "", action, post[0] if post else "")
        # append the guess into list
        guesses.append(list(guess))

        # if answer and guess is equal, increment correct_count
        if ans == guess:
            correct_count += 1

    # calculate precision, recall, f_measure scores
    precision = correct_count / (correct_count + len(test_sentences) - correct_count)
    recall = correct_count / (correct_count + 0)
    f_measure = 2 * precision * recall / (precision + recall)

    print(f"Correct:\t{correct_count} / {len(test_sentences)}")
    print(f"Wrong:\t{len(test_sentences) - correct_count} / {len(test_sentences)}")

    # print them to console
    print(f"Precision:\t{precision}")
    print(f"Recall:\t\t{recall}")
    print(f"F Mesaure:\t{f_measure}")

    # save output in csv file
    save_output("CS372_HW4_output_20190046.csv", test_sentences, guesses)
