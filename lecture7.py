import re
import nltk


if __name__ == "__main__":
    # raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
    # though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
    # well without--Maybe it's always pepper that makes people hot-tempered,'..."""

    # print(re.split(r" ", raw))
    # print(re.split(r"[ \t\n]", raw))
    # print(re.split(r"\W+", raw))
    # print(re.findall(r"\w+|\S\w*", raw))

    text = "That U.S.A. poster-print costs{$12.40..."
    pattern = r""" (?x)
        (?:[A-Z]\.)+
    |   \w+(?:-\w+)*
    |   \$\d+(?:\.\d+)?%?
    |   \.\.\.
    |   [][.,;"'?():-_`]
    """

    print(nltk.regexp_tokenize(text, pattern))
    print(nltk.regexp_tokenize(text, pattern, gaps=True))
