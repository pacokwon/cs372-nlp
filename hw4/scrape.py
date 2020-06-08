from bs4 import BeautifulSoup
import requests


def remove_newlines(l):
    return [e for e in l if e != "\n"]


with open("cites.tsv", "r") as file, open("newcites.tsv", "w") as newfile:
    cnt = 0
    for line in file:
        splitted = line.strip("\n").split("\t")
        url = splitted[0]
        PMID = url.split("/")[3]
        journal = splitted[1]
        year = splitted[2]
        soup = BeautifulSoup(requests.get(url).text)
        title = soup.select(".heading-title")[0].text.strip(" \n\t\r")
        affiliations = soup.select(".affiliations .item-list")
        if not affiliations or not remove_newlines(affiliations[0].contents):
            affiliation = "Unknown"
        else:
            affiliation = remove_newlines(affiliations[0].contents)[-1].text
        cnt += 1
        print(f"{cnt}: {PMID}\t{year}\t{journal}\t{affiliation}")
        print(f"{PMID}\t{year}\t{journal}\t{affiliation}", file=newfile)
