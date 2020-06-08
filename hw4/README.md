# Homework 4

There are 5 submission files excluding this README:
* `CS372_HW4_output_20190046.csv`
    this csv file contains 11 fields for each row:
    1. sentence
    2. answer triplet 1
    3. answer triplet 2
    4. answer triplet 3
    5. guessed triplet 1
    6. guessed triplet 2
    7. guessed triplet 3
    8. PMID
    9. year
    10. journal title
    11. organization
* `CS372_HW4_report_20190046.docx`
* `CS372_HW4_report_20190046.pdf`
* `CS372_HW4_code_20190046.py`
* `rand_input.csv`

## Execution
Only 2 files are required to run the python script:
* `CS372_HW4_code_20190046.py`
* `rand_input.csv`

`rand_input.csv` contains exactly 100 rows of collected sentences along with
the annotated triplet, PMID, year, journal title, organization for each row.
the orders are randomized and the last 20 rows were used to test the algorithm.

the python script assumes that `rand_input.csv` is in the same directory as
the script itself, so please keep that in mind

to execute script, run:
```bash
python CS372_HW4_code_20190046.py
```

Thank you!
