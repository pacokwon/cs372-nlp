# CS372 Homework 3 - 20190046 권해찬

The submission files are:
* `README.md`
* `requirements.txt`
* `CS372_HW3_output_20190046.csv`
* `CS372_HW3_code_20190046.py`
* `CS372_HW3_report_20190046.docx`
* `CS372_HW3_report_20190046.pdf`
* `corpora.zip`
* `html.zip`


## Requirements

The dependencies are listed in the requirements.txt file. The actual dependencies are not many, including BeautifulSoup and requests. However, it speicifes additional packages related to neovim only used during development, so please bear that in mind

To install dependencies:
```
pip install -r requirements.txt
```

This script needs COCA to collect sentences from the corpora. Make sure to uncompress `corpora.zip` and put the `corpora` directory in the same directory as the python script.

The `corpora` directory includes:
* `wlp_blog.txt`
* `wlp_fic.txt`
* `wlp_news.txt`
* `wlp_tvm.txt`

## Notes
The python script saves various html and json files to reduce time. Upon first execution, the script has to fetch everything from the web, so it will take some time to finish (roughly 1 ~ 1.5hr on my machine). With local file caching, it took about 5 minutes to complete.

The cache files include:
* `./html/` directory: will contain html files (about 8700 of them)
* `./crawled/` directory: will contain json files storing information of crawled sentences from the particular corpus
* `./heteronyms_list.json`: will contain user-collected heteronyms list
* `./tagger.pkl`: pickled BigramTagger file

These files will be created upon the first execution of the python script. Since the html files take really long to collect, I will include them as a compressed file as well just in case. If you want to use them, put the uncompressed `html` directory in the same directory as the script

Thank You.
