# IRWA project: fashion products search

Course project (Information Retrieval and Web Analytics). The corpus is an e-commerce fashion dataset
(28,080 products). `validation_labels.csv` has the relevance labels (1 relevant, 0 not) of 40 products for
two reference queries, used to choose and later evaluate the design:
- query 1: `women full sleeve sweatshirt cotton`
- query 2: `men slim jeans blue`

The statement of the assignment is `Part1/IRWA_project_part_1 (2).pdf`.

## Status

**Part 1: finished.** Data preparation (items 1 to 4, report sections 1 to 3) and exploratory data analysis
(`EDA.py`, report section 4) are done. `Part1/Report_Part1.pdf` is built from `Report_Part1.md` and is the only
thing handed in; it links to this repo.

Next (Part 2, indexing and ranking): the proposals of report section 4.7 are not applied yet (recover the full
brand from the URL, drop the `Style Code` value, demote out-of-stock products, log/rank-scale the price).

## Layout

```
validation_labels.csv                  labels of the two reference queries
Part1/
  fashion_products_dataset.json        the dataset
  DataPreparation.py                   preprocessing (import it, do not duplicate its logic)
  FieldSelection.py                    evaluates which fields to index -> field_selection_results/*.csv
  DataChecks.py                        figures quoted in the report    -> data_checks_results/*.csv
  EDA.py                               exploratory data analysis       -> eda_results/*.csv, eda_results/figures/*.png
  Report_Part1.md                      the report (source of the PDF)
  build_report.py                      Report_Part1.md -> Report_Part1.pdf (pandoc + XeLaTeX)
  Report_Part1.pdf                     the deliverable (short: setup and file map live in README.md)
README.md                              how to run, and which file backs each report section
```

## How to run

- **Run every script from `Part1/`.** The JSON is opened with a relative path on purpose (the file lives in `Part1/`).
- Needs `pandas`, `nltk`, `matplotlib`, `wordcloud`, `spacy` (+ `py -m spacy download en_core_web_sm`) and, for
  the PDF, `pypandoc_binary` (bundles pandoc) and a XeLaTeX install (MiKTeX). The NLTK stop words are downloaded
  on import. On the Windows machine everything is installed in `py` (Python 3.14).
- `python DataPreparation.py` only loads the data and prints a preview. `python FieldSelection.py` takes about
  a minute (it preprocesses the 28,080 products and evaluates 127 field combinations). `python EDA.py` takes
  about 2.5 minutes (spaCy over the 16,930 descriptions). `python build_report.py` builds the PDF.
- `FieldSelection.py` reads `validation_labels.csv` from the folder above `Part1/`.

## What `DataPreparation.py` does

- `normalize_string`: Unicode NFKD, strip diacritics, lowercase, join `t-shirt` into `tshirt`, keep word
  characters only (punctuation acts as a separator).
- `preprocess_text`: the above, then NLTK English stop words, then the Snowball stemmer.
- `preprocess(db_fashion)` returns, for each product:
  `{"pid", "original", "processed", "numeric"}`.
  - `original`: the untouched record. **Results shown to the user always come from here.**
  - `processed`: one token list per candidate field plus `text`, the merged tokens of `INDEXED_FIELDS`.
  - `numeric`: `selling_price`, `actual_price`, `discount` (percent), `average_rating`, `out_of_stock`, kept
    for the ranking stage. `None` when missing (not imputed). Not scaled yet.

## Decisions already taken (change them only with evidence)

- **Fields:** `title`, `description`, `sub_category`, `brand` and `product_details` are merged into one indexed
  text (`INDEXED_FIELDS`). `category` (96.6% the same value) and `seller` are not indexed but stay in `original`.
  Of `product_details` only the values are kept, not the names.
- **No field weights and no learned weights.** There are only 40 labeled products and 2 queries, and the same
  labels will be used for evaluation later, so fitting on them would inflate the results. The fields are
  chosen on/off, justified by the evidence in `field_selection_results/`. No logistic regression.
- **Numeric fields are not indexed as text.** They are ranking signals (section 3 of the report).
- **No Presidio / anonymization.** It would erase brand and seller names, which are search terms. Entity
  recognition belongs only to the exploratory analysis.
- **`t-shirt` is joined** because the NLTK stop-word list contains the letter `t` and `T-Shirt` became `shirt`.

## Known limitations (documented in the report unless noted)

- The NLTK stop-word list also removes `no` and `not`, so negations and `Reversible = No` values are lost.
- Only 40 labeled products and 2 queries: AUC differences of a few thousandths are noise. The fields
  `category`, `sub_category`, `brand` and `seller` never contain a query word, so the labels cannot judge them.
  They were decided by reasoning.
- The labeled products look like top results of an earlier retrieval, so they all match the title. This
  understates the value of `title`.
- Found by the EDA (section 4.4/4.2): `brand` is cut off in 88.1% of the products (`Pu` = puma; the full name is
  at the start of the URL path), and an empty `seller` means out of stock. 94.5% of the `product_details` terms
  seen once come from `Style Code`.
- Not in the report: the preprocessed products are not saved to disk, `DataPreparation.py` prints debug output
  on import (the DataFrame preview and `FIELDS`), and `FieldSelection.py` expects `INDEXED_FIELDS` to keep the
  same order as `FIELDS` (otherwise a `KeyError`).

## Workflow

1. Change the preprocessing or the fields in `DataPreparation.py`.
2. Rerun `python FieldSelection.py`, `python DataChecks.py` and `python EDA.py` from `Part1/`. The CSV files are
   overwritten.
3. Check that the figures in `Part1/Report_Part1.md` still match the CSV files. Every figure cites its file.
4. Rebuild the PDF with `python build_report.py` and look at the pages.

| Report section | Evidence |
|---|---|
| 1.1 (t-shirt) | `data_checks_results/preprocessing_checks.csv` |
| 2.2 field profile | `field_selection_results/field_profile.csv` |
| 2.3 and 2.4 evaluation | `ablation.csv`, `combinations.csv`, `rankings.csv`, `field_values.csv`, `query_term_matches.csv` |
| 3 numeric fields | `data_checks_results/numeric_fields.csv` |
| 4 exploratory analysis | `eda_results/*.csv` (each table names its file), figures in `eda_results/figures/` |

Do not edit the CSV files by hand; they are generated.

## Style

- Plain, short Python: small functions with a one-line docstring, no frameworks, no premature options. The team
  asked to keep it simple ("chill") after an over-engineered start.
- The report is written in English for the instructors. Keep claims that the data cannot prove marked as
  reasoning, not as evidence.
- Results go to CSV files and the report cites them, instead of generating a Markdown report from code.
