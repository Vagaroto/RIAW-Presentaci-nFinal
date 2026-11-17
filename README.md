# IRWA project: fashion products search

Course project of Information Retrieval and Web Analytics. The corpus is an e-commerce fashion dataset
(28,080 products from Flipkart). The deliverable of Part 1 is the report `Part1/Report_Part1.pdf`.

## Layout

```
validation_labels.csv                  relevance labels (1/0) of 40 products for the two reference queries
Part1/
  fashion_products_dataset.json        the dataset
  IRWA_project_part_1 (2).pdf          statement of Part 1
  DataPreparation.py                   text preprocessing, fields to index, numeric values (imported by the others)
  FieldSelection.py                    BM25 + AUC for every combination of fields -> field_selection_results/
  DataChecks.py                        numeric fields and t-shirt check           -> data_checks_results/
  EDA.py                               exploratory data analysis                  -> eda_results/ (+ figures/)
  Report_Part1.md                      source of the report
  build_report.py                      Report_Part1.md -> Report_Part1.pdf
  Report_Part1.pdf                     the report
```

## How to run

Run every script from the `Part1/` folder (the dataset is opened with a relative path).

```bash
pip install pandas nltk matplotlib wordcloud spacy pypandoc_binary
python -m spacy download en_core_web_sm

cd Part1
python FieldSelection.py   # ~1 min
python DataChecks.py       # ~20 s
python EDA.py              # ~2.5 min (spaCy over 16,930 descriptions)
python build_report.py     # needs a LaTeX install with XeLaTeX (e.g. MiKTeX, TeX Live)
```

The CSV and PNG files are generated; they are overwritten on every run and should not be edited by hand.

## Where each number of the report comes from

| Report section | Evidence |
|---|---|
| 1.1 Text processing (t-shirt step) | `data_checks_results/preprocessing_checks.csv` |
| 2.2 Distinctiveness of each field | `field_selection_results/field_profile.csv` |
| 2.3–2.4 Evaluation of the fields | `field_selection_results/ablation.csv`, `combinations.csv`, `rankings.csv`, `field_values.csv`, `query_term_matches.csv` |
| 3 Numeric fields | `data_checks_results/numeric_fields.csv` |
| 4 Exploratory data analysis | `eda_results/*.csv` (each table of the report names its file), figures in `eda_results/figures/` |

## Preprocessing in short

- `normalize_string`: Unicode NFKD, strip diacritics, lowercase, join `t-shirt` into `tshirt`, keep word characters.
- `preprocess_text`: the above, then NLTK English stop words, then the Snowball stemmer.
- `preprocess(db_fashion)` returns, per product, `{"pid", "original", "processed", "numeric"}`. Results shown to
  the user always come from `original`.
