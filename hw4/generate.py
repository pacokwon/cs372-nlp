import csv
from nltk.tag import StanfordNERTagger
from nltk import word_tokenize
from pprint import pprint
from random import randint, shuffle

# tagger = StanfordNERTagger(
#     "./trained-model.ser.gz", "./resources/stanford-ner-4.0.0/stanford-ner.jar",
# )

# with open("./training/testing.txt", "r") as file:
#     lines = [line.rstrip("\n") for idx, line in enumerate(file) if idx % 2 == 0]

# few = [tagger.tag(word_tokenize(line)) for line in lines]


def generate(num, total):
    num_list = []
    while len(num_list) < num:
        ri = randint(0, total - 1)

        if ri in num_list:
            continue

        num_list.append(ri)

    return sorted(num_list)


# with open("sents", "r") as file1, open("annotated.csv", "r") as file2:
#     lines = [line.rstrip("\n") for line in file1]
#     annotated = [line.rstrip("\n") for line in file2]


# with open("training.txt", "w") as training, open("testing.txt", "w") as testing:
#     training_idx = generate(80, 102)
#     cursor = 0

#     for idx, (line, ann) in enumerate(zip(lines, annotated)):
#         if idx == training_idx[cursor]:
#             print(line, file=training)
#             print(ann, file=training)
#             cursor += 1
#         else:
#             print(line, file=testing)
#             print(ann, file=testing)

if __name__ == "__main__":
    with open("./input.csv") as file:
        reader = csv.reader(file, delimiter=",", quotechar='"')
        rows = [row for row in reader]

    with open("rand_input.csv", "w") as file:
        shuffle(rows)
        writer = csv.writer(file, delimiter=",", quotechar='"')

        for row in rows:
            writer.writerow(row)
