from bs4 import BeautifulSoup
import csv
import itertools
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import os.path
import re
import requests
from urllib.parse import urlparse

DEBUG = False


class Queue:
    """
    Minimal Queue class implementation for breadth first search usage
    """

    def __init__(self):
        """
        Queue class constructor
        """
        self.queue = []

    def is_empty(self):
        """
        Return whether the queue is empty or not

        :returns: whether the queue is empty or not
        :rtype: bool
        """
        return len(self.queue) == 0

    def enqueue(self, item):
        """
        Insert the given item into the queue
        """
        self.queue.append(item)

    def dequeue(self):
        """
        Pop and return the item in the queue

        :returns: front item in the queue
        :rtype: any
        """
        if self.is_empty():
            raise IndexError("Queue is empty!")

        return self.queue.pop(0)


def get_word_index(text, word, offset):
    """
    Get the "index" of `word` in a given `text` according to `offset`

    the "index" indicates the # of occurrences of the same word before the specified word

    e.g. index of word "Hello"
    Hello world Hello
    ^ 0         ^ 1

    :param text: the whole text to find words from
    :type text: str
    :param word: the word to find the index of
    :type word: str
    :param offset: index of the first letter of word in the `text`
    :type offset: int
    :returns: "index" of the given word
    :rtype: int
    """
    indices = [
        m.start() for m in re.finditer(rf"(?<![A-Za-z]){word}(?![A-Za-z])", text)
    ]
    return indices.index(offset)


def get_word_path(trees, word, word_index):
    """
    Return the path to a word, whose position is specified by `word_index`, in the tree

    `word_index` describes the same thing as described by "index" in the function get_word_index

    :param trees: list of trees, each of which are parsed sentences
    :type trees: list[nltk.Tree]
    :param word: word to find the path of from the tree
    :type word: str
    :param word_index: # of occurrences of the same word before the specified word
    :type word_index: int
    :returns: path to `word` in the tree
    :rtype: tuple[int]
    """
    count = 0
    word_path = []

    tree = nltk.Tree("root", trees)

    # helper function dfs
    def dfs(tree, path):
        nonlocal word_path, count

        if word_path:
            return

        if type(tree) == tuple:
            if tree[0] != word:
                return

            if count == word_index:
                word_path = tuple(path)
            else:
                count += 1
        else:
            for idx, subtree in enumerate(tree):
                dfs(subtree, path + [idx])

    dfs(tree, [])
    return word_path


def parse_data(filename):
    """
    Return a list of python dictionaries containing information for each row
    in the specified tsv file

    Each element in the list contains the following keys in string format:
    ID, Text, Pronoun, Pronoun-offset, A, A-offset, A-coref, B, B-offset, B-coref, URL

    :param filename: filename of tsv file to be read
    :type filename: str
    :returns: list of dictionaries containing each row's information
    :rtype: list of dictionaries
    """
    strip = lambda l: [e.strip("\n\r\t") for e in l]
    intify = lambda l: [int(e) if e.isnumeric() else e for e in l]

    with open(filename, "r") as file:
        lines = [intify(strip(line.split("\t"))) for line in file]

    columns = lines[0]
    return [dict(zip(columns, line)) for line in lines[1:]]


def chunked_sentences(sentences, parser):
    """
    POS tag and parse sentences with the given parser

    :param sentences: given sentences in a single str object
    :type sentences: str
    :param parser: nltk.RegexpParser object with custom grammar
    :type parser: nltk.RegexpParser
    """
    pos_tagged_sentences = [
        pos_tag(word_tokenize(sent)) for sent in sent_tokenize(sentences)
    ]

    chunked_sentences = [
        parser.parse(pos_tagged_sent) for pos_tagged_sent in pos_tagged_sentences
    ]

    return chunked_sentences


def tag_data(text_list, parser):
    """
    Tag every text in the given `text_list` and return a list of tagged texts

    The given text_list must be a list of strings, where each text consists of one or more sentences
    This function:
        splits each text into sentences,
        pos_tags each word in each sentences,
        then chunks each sentence to group noun phrases
    and then returns the result

    :param text_list: list of texts, where each text can consist of one or more sentences in string
    :type text_list: list of str
    :returns: list of tagged texts
    :rtype: list[list[Tree]]
    """

    return [chunked_sentences(sents, parser) for sents in text_list]


def validate_np(pronoun, encountered, proposed_np):
    """
    If there is an NP in the middle of a proposed_np and pronoun return True
    otherwise, return False

    :param pronoun: path to the pronoun that one desires to resolve
    :type pronoun: tuple[int]
    :param encountered: list of paths to encountered noun phrases
    :type encountered: list[tuple[int]]
    :param proposed_np: noun phrase to see if there is an NP in the middle between it and the prnoun
    :type proposed_np: tuple[int]
    :returns: whether or not there is an NP in the middle of a proposed_np and pronoun
    :rtype: bool
    """
    for np in encountered:
        if np != pronoun and np != proposed_np and proposed_np < np:
            return True
    return False


def collect_noun_phrase_bfs(tree):
    """
    Collect noun phrase in a left-to-right, breadth-first fashion, then
    return the collected noun phrases

    :param tree: tree to be scanned
    :type tree: nltk.Tree
    :returns: list of paths to collected noun phrases
    :rtype: list[tuple[int]]
    """
    queue = Queue()

    # initialize queue contents to child. not root
    for idx in range(len(tree[()])):
        queue.enqueue((idx,))

    encountered = []
    while not queue.is_empty():
        popped = queue.dequeue()

        if type(tree[popped]) != nltk.Tree:
            continue

        if tree[popped].label() in ["NP", "S"]:
            encountered.append(popped)

        for idx in range(len(tree[popped])):
            queue.enqueue((*popped, idx))

    return encountered


def prev_sents_antecedents(trees):
    """
    Collect and return proposed antecedents from previous sentences
    the antecedents are sorted by priority in descending order.

    :param trees: trees of previous sentences
    :type trees: list[nltk.Tree]
    :returns: list of paths to proposed antecedents
    :rtype: list[tuple[int]]
    """
    antecedents_list = []
    for idx, tree in enumerate(trees):
        nps = collect_noun_phrase_bfs(tree)
        antecedents_list.append([(idx, *np) for np in nps])

    # reverse list, flatten, and return
    return list(itertools.chain(*reversed(antecedents_list)))


def is_nominal(label):
    """
    A function to tell if a UPenn POS label is a nominal group or not

    :param label: UPenn POS label
    :type label: str
    :returns: True if label is nominal group. False otherwise
    :rtype: bool

    """
    return label.startswith("NN")


def hobbs(sents, pronoun_path):
    """
    Implementation of Hobb's Algorithm

    Accept a list of sentences, which are individually nltk.Tree objects
    and resolve the pronoun denoted by `pronoun_path`

    The full algorithm is as follows:
    1. Begin at the NP node immediately dominating the pronoun.
    2. Go up the tree to the first NP or S node encountered.
       Call this node X and call the path used to reach it p.
    3. Traverse all branches below node X to the left of path p in a left-to-right, breadth-first fashion.
       Propose as an antecedent any NP node that is encountered which has an NP or S node between it and X.
    4. If node X is the highest S node in the sentence, traverse the surface parse trees of previous sentences
       in the text in order of recency, the most recent first; each tree is traversed in a left-to-right, breadth-first manner,
       and when an NP node is encountered, it is proposed as an antecedent.
       If X is not the highest S node in the sentence, continue to step 5.
    5. From node X, go up the tree to the first NP or S node encountered.
       Call this new node X, and call the path traversed to reach it p.
    6. If X is an NP node and if the path p to X did not pass through the Nominal node that X immediately dominates, propose X as the antecedent.
    7. Traverse all branches below node X to the left of path p in a left-to-right, breadth-first manner.
       Propose any NP node encountered as the antecedent.
    8. If X is an S node, traverse all the branches of node X to the right of path p in a left-to-right, breadth-first manner,
       but do not go below any NP or S node encountered. Propose any NP node encountered as the antecedent.
    9. Go to step 4.

    :param sents: list of sentences
    :type sents: list[nltk.Tree]
    :param pronoun_path: tuple containing path to pronoun
    :type pronoun_path: tuple[int]
    :returns: path to resolved word
    :rtype: tuple[int]
    """
    # the first element of pronoun_path denotes the index of the sentence
    sent_idx = pronoun_path[0]
    sent = sents[sent_idx]
    path = pronoun_path[1:]  # the rest expresses the path within the sentence tree
    if DEBUG:
        print("Sentence: ")
        print(" ".join(list(map(lambda x: x[0], sent.leaves()))))
        print("Tree: ")
        print(sent)
        print("Path: ", path)

    # 1. Begin at the NP node immediately dominating the pronoun
    np_dominating = path[:-1]

    # 2. Go up the tree to the first NP or S node encountered. Call this node X and the path used to reach it p.
    X_to_dom = [np_dominating[-1]]
    X = np_dominating[:-1]
    while not (sent[X].label() == "NP" or sent[X].label() == "S"):
        X_to_dom.append(X[-1])
        X = X[:-1]
    X_to_dom = tuple(reversed(X_to_dom))  # contains path to the dominating NP
    if DEBUG:
        print("=============== Step 2 ===============")
        print("X: ", X)
        print("X_TO_DOM: ", X_to_dom)
        print("======================================")

    # 3. Traverse all branches below node X to the left of path p in a left-to-right, breadth-first fashion.
    #    Propose as an antecedent any NP node that is encountered which has an NP or S node between it and X.
    np_encountered_raw = collect_noun_phrase_bfs(sent[X])
    if DEBUG:
        print(np_encountered_raw)
    np_encountered = [
        (*X, *np_path) for np_path in np_encountered_raw if np_path[0] < X_to_dom[0]
    ]

    for np in np_encountered:
        # np contains path to the NP phrase
        # asking if np is valid
        if validate_np(path, np_encountered, np):
            return (sent_idx, *np)

    while True:
        if DEBUG:
            print("Loop")
        # 4. If node X is the highest S node in the sentence, traverse the surface parse trees of previous sentences
        #    in the text in order of recency, the most recent first; each tree is traversed in a left-to-right, breadth-first manner,
        #    and when an NP node is encountered, it is proposed as an antecedent.
        #    If X is not the highest S node in the sentence, continue to step 5.
        if X == ():
            proposed_antecedents = prev_sents_antecedents(sents[:sent_idx])
            if DEBUG:
                print("=============== Step 4 ===============")
                print(sent_idx)
                print(sents[:sent_idx])
                print("Proposed: ")
                print(proposed_antecedents)
                print("======================================\n")
            if proposed_antecedents:
                return proposed_antecedents[0]
            else:
                return None

        # 5. From node X, go up the tree to the first NP or S node encountered. Call this new node X, and call the path traversed to reach it p.
        new_to_old = [X[-1]]  # stores the relative path from new X to old X
        X = X[:-1]  # stores the absolute path to new X
        while not (sent[X].label() == "NP" or sent[X].label() == "S"):
            new_to_old.append(X[-1])
            X = X[:-1]
        new_to_old = tuple(reversed(new_to_old))

        if DEBUG:
            print("=============== Step 5 ===============")
            print("New X: ")
            print(X)
            print("New to Old: ")
            print((*X, *new_to_old))
            print("======================================\n")

        # 6. If X is an NP node and if the path p to X did not pass through the Nominal node that X immediately dominates, propose X as the antecedent.
        if DEBUG:
            print("=============== Step 6 ===============")
            print("======================================\n")

        if sent[X].label() == "NP" and not is_nominal(sent[X][new_to_old[0]].label()):
            return (sent_idx, *X)

        # 7.Traverse all branches below node X to the left of path p in a left-to-right, breadth-first manner.
        #   Propose any NP node encountered as the antecedent.
        np_encountered_raw = collect_noun_phrase_bfs(sent[X])
        np_encountered = [
            (*X, *np_path)
            for np_path in np_encountered_raw
            if np_path[0] < new_to_old[0]
        ]
        if DEBUG:
            print("=============== Step 7 ===============")
            print("NP Encountered: ")
            print(np_encountered)
            print("======================================\n")
        if np_encountered:
            return (sent_idx, *np_encountered[0])

        # 8. If X is an S node, traverse all the branches of node X to the right of path p in a left-to-right, breadth-first manner,
        #    but do not go below any NP or S node encountered. Propose any NP node encountered as the antecedent.
        if sent[X].label() == "S":
            queue = Queue()

            for idx in range(len(sent[X])):
                queue.enqueue((*X, idx))

            np_encountered_raw = []
            while not queue.is_empty():
                popped = queue.dequeue()

                if type(sent[popped]) != nltk.Tree:
                    continue

                if sent[popped].label() in ["NP", "S"]:
                    np_encountered_raw.append(popped)
                else:
                    for idx in range(len(sent[popped])):
                        queue.enqueue((*popped, idx))

            np_encountered = [
                (*X, *np_path)
                for np_path in np_encountered_raw
                if np_path[0] > new_to_old[0]
            ]

            if DEBUG:
                print("=============== Step 8 ===============")
                print("NP Raw: ")
                print(np_encountered_raw)
                print("NP Encountered: ")
                print(np_encountered)
                print("======================================\n")

            if np_encountered:
                return (sent_idx, *np_encountered[0])


def subtree(trees, path):
    """
    Print the contents of a subtree, specified the `path` parameter

    :param trees: list of trees, each of which are parsed sentences
    :type trees: list[nltk.Tree]
    :param path: path to subtree
    :type path: tuple[int]
    :returns: leaves of subtree
    :rtype: str
    """

    stree = trees[path[0]][path[1:]]
    if type(stree) == tuple:
        return stree[0]
    else:
        return " ".join(map(lambda x: x[0], stree.leaves()))


def save_as_tsv(filename, data):
    """
    Save the given data to a tsv file

    :param filename: name of file to be saved to
    :type filename: str
    :param data: list of rows to be saved to the file
    :type data: list[list]
    """
    with open(filename, "w") as file:
        writer = csv.writer(file, delimiter="\t", quotechar='"')
        for row in data:
            writer.writerow(row)


def snippet_guess(datum, parser):
    """
    Predict and return snippet-context guess results

    Since this is snippet-context guessing, we can only look at the text from
    the test data. Apply hobbs algorithm for pronoun resolution to given
    text and pronoun.

    :param datum: single row of test data parsed in a dictionary
    :type datum: dict
    :param parser: nltk.RegexpParser object with custom grammar
    :type parser: nltk.RegexpParser
    :returns: list that contains values of required fields (id, A-coref, B-coref)
    :rtype: list
    """
    word_index = get_word_index(
        datum["Text"], datum["Pronoun"], datum["Pronoun-offset"]
    )
    chunked = chunked_sentences(datum["Text"], parser)
    path = get_word_path(chunked, datum["Pronoun"], word_index)

    if not path:
        print("Snippet Guess: No Path!")
        return [datum["ID"], False, False]

    result = hobbs(chunked, path)

    if result == None:
        print("Snippet Guess: Result is None!")
        return [datum["ID"], False, False]

    snippet_guess = subtree(chunked, result)
    return [datum["ID"], snippet_guess == datum["A"], snippet_guess == datum["B"]]


def process(sent):
    """
    Preprocess a sentence so that characters are correctly interpreted

    :param sent: sentence to run the preprocess on
    :type sent: str
    :returns: preprocessed sentence
    :rtype: str
    """
    return sent.replace("\n", "").replace("``", "“").replace("''", "”")


def find_from_paragraphs(paragraphs, sentence):
    """
    Return the index of paragraph that contain a given sentence

    :param paragraphs: list of paragraphs
    :type paragraphs: list[str]
    :param sentence: sentence to look for
    :type sentence: str
    :returns: the index of paragraph that contains `sentence`
    :rtype: int
    """
    idx = 0
    while idx < len(paragraphs):
        if sentence[: (len(sentence) // 2)] in paragraphs[idx]:
            return idx
        idx += 1
    return -1


def count_word(text, word):
    """
    Count the actual occurrences of a word in a given text

    Using the count function in str also counts occurrences that just
    happen to be embedded in a completely different word

    :param text: the text to search `word` from
    :type text: str
    :param word: the word to search in text
    :type word: str
    :returns: the occurrences of `word` in text
    :rtype: int
    """
    return sum(1 for _ in re.finditer(rf"(?<![A-Za-z]){word}(?![A-Za-z])", text))


def get_related_text(text, url, pronoun, word_index):
    """
    Given a text(which originally is a portion of a Wikipedia page),
    extract surrounding sentences, which we call "context", and return it

    :param text: given text from the test dataset
    :type text: str
    :param url: url to the wikipedia page that `text` came from
    :type url: str
    :param pronoun: the target pronoun to resolve
    :type pronoun: str
    :param word_index: the "index" of `pronoun`. refer to get_word_index
    :type word_index: int
    :returns: two-element tuple, (context, word_index)
              where the first element is a list containing the sentences in the new context,
              and the second element is the new "index" of the given pronoun in the context
    :rtype: tuple(list, int)
    """
    url = urlparse(url)
    topic = url.path.split("/")[-1]

    sents = sent_tokenize(text)

    sent_idx = -1
    count = 0
    while count <= word_index:
        sent_idx += 1
        count += count_word(sents[sent_idx], pronoun)

    sent_with_pronoun = process(sents[sent_idx])

    path = "./html"
    if not os.path.exists(path):
        os.mkdir(path)

    filename = f"{path}/{topic}.html"
    # construct soup
    if os.path.exists(filename):
        with open(filename) as fp:
            soup = BeautifulSoup(fp, features="lxml")
    else:
        html_content = process(requests.get(url.geturl()).text)
        with open(filename, "w") as fp:
            fp.write(html_content)
        soup = BeautifulSoup(html_content, features="lxml")

    # print(soup)
    paragraphs = [
        re.sub(r"\[\d+\]", "", p.text) for p in soup.select(".mw-parser-output p")
    ]
    pg_idx = find_from_paragraphs(paragraphs, sent_with_pronoun)

    if pg_idx == -1:
        return (text, word_index)

    paragraph = sent_tokenize(paragraphs[pg_idx])

    idx = 0
    while idx < len(paragraph):
        if sent_with_pronoun[: (len(sent_with_pronoun) // 2)] in paragraph[idx]:
            break
        idx += 1
    if idx == len(paragraph):
        return (text, word_index)

    new_context = []
    new_word_index = word_index - (count - count_word(sents[sent_idx], pronoun))

    for i in range(idx):
        new_context.append(paragraph[i])
        new_word_index += count_word(paragraph[i], pronoun)

    new_context.append(sents[sent_idx])

    return " ".join(new_context), new_word_index


def page_guess(datum, parser):
    """
    Predict and return page-context guess results

    Since page-context allows us to look at the Wikipedia page,
    we extract context from the page and make predictions based on
    that text, using hobb's algorithm for pronoun resolution.
    The context is the only different factor from the snippet_guess function's logic

    :param datum: single row of test data parsed in a dictionary
    :type datum: dict
    :param parser: nltk.RegexpParser object with custom grammar
    :type parser: nltk.RegexpParser
    :returns: list that contains values of required fields (id, A-coref, B-coref)
    :rtype: list
    """
    word_index = get_word_index(
        datum["Text"], datum["Pronoun"], datum["Pronoun-offset"],
    )

    context, new_word_index = get_related_text(
        datum["Text"], datum["URL"], datum["Pronoun"], word_index
    )

    chunked = chunked_sentences(context, parser)
    path = get_word_path(chunked, datum["Pronoun"], new_word_index)

    if not path:
        print("Page Guess: No Path!")
        return [datum["ID"], False, False]

    result = hobbs(chunked, path)

    if result == None:
        print("Page Guess: Result is None!")
        return [datum["ID"], False, False]

    page_guess = subtree(chunked, result)
    return [datum["ID"], page_guess == datum["A"], page_guess == datum["B"]]


grammar = r"""
    NP: {<PRP\$?>}
        {<NN.*>+}
    PP: {<IN><NP>}
    VP: {<VB.*><NP>}
    S: {<NP><VP>}
"""

cp = nltk.RegexpParser(grammar)
data = parse_data("./gap-test.tsv")

snippet_guesses = []
page_guesses = []

for idx, datum in enumerate(data):
    print(f"Idx: {idx}")

    snippet_guess_row = snippet_guess(datum, cp)
    snippet_guesses.append(snippet_guess_row)

    page_guess_row = page_guess(datum, cp)
    page_guesses.append(page_guess_row)

save_as_tsv("snippet_output.tsv", snippet_guesses)
save_as_tsv("page_output.tsv", page_guesses)
