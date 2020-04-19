import nltk


if __name__ == "__main__":
    # u = chr(40960) + "abcd" + chr(1972)  # 'ꀀabcd\u07b4'
    # encoded = u.encode("utf-8")
    # print(encoded)  # b'\xea\x80\x80abcd\xde\xb4'

    path = nltk.data.find("corpora/unicode_samples/polish-lat2.txt")

    f = open(path, encoding="latin2")
    for line in f:
        line = line.strip()
        print(line.encode("raw_unicode_escape"))
        print(line)
