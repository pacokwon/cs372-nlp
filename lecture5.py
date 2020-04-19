# from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
from urllib import request
from nltk import word_tokenize
import nltk

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode("utf8")
raw = BeautifulSoup(html, "lxml").get_text()
tokens = word_tokenize(raw)
text = nltk.Text(tokens)
text.concordance("gene")
