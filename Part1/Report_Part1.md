# IRWA Project – Part 1: Text processing and exploratory data analysis

> Draft. Sections 1 to 3 (data preparation) are complete. Section 4 (exploratory data analysis) is not written yet. Every figure comes from the CSV files listed in the Repository section.

## Repository

Code and results: **TODO: GitHub link** (folder `RIAW-Presentaci-nFinal/Part1/`). The scripts are run from the `Part1/` folder. The figures in this report can be checked in these files:

| File or folder | Content |
|---|---|
| `Part1/DataPreparation.py` | Text processing, the fields to index (`INDEXED_FIELDS`), the numeric values, and the `preprocess` function that builds the object of each product. |
| `Part1/FieldSelection.py` | Ranks the labeled products with BM25 for every combination of fields and saves the results in `Part1/field_selection_results/`. |
| `Part1/field_selection_results/` | `field_profile.csv`, `query_term_matches.csv`, `field_values.csv`, `ablation.csv`, `combinations.csv` and `rankings.csv`. |
| `Part1/DataChecks.py` | Saves the figures about the numeric fields and the `t-shirt` step in `Part1/data_checks_results/`. |
| `Part1/data_checks_results/` | `numeric_fields.csv` and `preprocessing_checks.csv`. |
| `Part1/fashion_products_dataset.json` | The dataset. |
| `validation_labels.csv` | The relevance labels of the two reference queries. |

## 1. Data preparation

### 1.1 Text processing (`title`, `description` and the other indexed text fields)

Every text field goes through the same steps (`preprocess_text` in `DataPreparation.py`):

1. **Unicode normalization and diacritics removal** (NFKD, combining marks dropped), so `"Azúl"` and `"azul"` are the same term.
2. **Lowercasing.**
3. **Joining `t-shirt`:** `t-shirt` and `t shirt` become the single word `tshirt` (see below).
4. **Tokenization and punctuation removal:** the text is split into runs of word characters, so every punctuation mark acts as a separator. This also handles missing spaces such as `"look.Designed"` or `"Comfort,Skin"`.
5. **Stop-word removal** with the English list of NLTK.
6. **Stemming** with the Snowball stemmer of NLTK.

Steps 1 and 3 go beyond the required list (the "anything else" of the assignment). Step 3 is needed because the NLTK stop-word list contains the single letter `t`: without it, `"T-Shirt"` is reduced to `shirt` and cannot be told apart from a plain `"Shirt"`, which is a different product. 12,247 of the 28,080 titles contain `T-Shirt`, so the effect on retrieval would be large (`data_checks_results/preprocessing_checks.csv`). `Sweatshirt` is not affected, because it is already a single word.

Examples:

| Original | Processed tokens |
|---|---|
| `Solid Women Multicolor Track Pants` | solid, women, multicolor, track, pant |
| `Typography Women V Neck Multicolor T-Shirt  (Pack of 2)` | typographi, women, v, neck, multicolor, tshirt, pack, 2 |
| `Yorker trackpants made from 100% rich combed cotton giving it a rich look.Designed for Comfort,Skin friendly fabric,itch-free waistband & great for all year round use Proudly made in India` | yorker, trackpant, made, 100, rich, comb, cotton, give, rich, look, design, comfort, skin, friend, fabric, itch, free, waistband, great, year, round, use, proud, made, india |

Known limitation: the NLTK stop-word list also removes `no` and `not`, so negations are lost and values such as `Reversible = No` disappear from `product_details`. This was not changed.

Each product is stored as an object with four parts: the **original** record, untouched; the **processed** tokens (one list per field and the merged `text` that is indexed); the `numeric` values (section 3); and the `pid`. Retrieval works on the processed tokens, but the results are always returned from the original record.

### 1.2 Fields returned by a query

`pid`, `title`, `description`, `brand`, `category`, `sub_category`, `product_details`, `seller`, `out_of_stock`, `selling_price`, `discount`, `actual_price`, `average_rating` and `url` are taken from the original record, so none of them is affected by the preprocessing. `pid` is never modified because it is used in the evaluation.

## 2. Handling of `category`, `sub_category`, `brand`, `product_details` and `seller`

### 2.1 Decision

The fields `title`, `description`, `sub_category`, `brand` and `product_details` are **merged into one text field that is indexed**, without field weights. `category` and `seller` are **not indexed**, but they remain in the original record and are returned with the results. For `product_details`, which is a list of `{name: value}` pairs, only the values are kept (`Fit: Slim` contributes `slim`). The names are repeated in thousands of products and would only add noise.

### 2.2 Approaches considered

| Approach | Pros | Cons |
|---|---|---|
| **One merged text field** (chosen) | Simple index and a single ranking function. No weights to tune. A query term matches wherever it appears. | The origin of a term is lost (a word in the brand is the same as the word in the description). Long fields (`product_details` has 24 tokens on average) dilute short and distinctive ones (`brand`, `sub_category`). The importance of a field cannot be changed. |
| **Separate fields in the inverted index** | Each field can be searched or boosted alone. Short distinctive fields are not diluted by long ones. | It needs a way to combine the fields, normally weights, and those need data to be set. Here there are only 40 labeled products and 2 queries, too few to estimate them reliably. Sparse fields give no score (`description` is empty for 39.7% of the products, `brand` for 8.0%). More complexity and a larger index. |
| **Fields with few values as filters or facets** (alternative) | Exact results for values like `sub_category`. | The query would have to be translated into a filter, which a free-text search does not do. |

The merged field was chosen because the available evidence (section 2.4) is enough to decide which fields to use but not to estimate how much each one should weigh.

### 2.3 Distinctiveness of each field

| Field | Empty | Distinct values | Most common value | Comment |
|---|---|---|---|---|
| `category` | 0.0% | 4 | Clothing and Accessories (96.6%) | Almost constant: it does not separate products. |
| `sub_category` | 0.0% | 24 | Topwear (54.5%) | Few values: low distinctiveness alone. |
| `brand` | 8.0% | 324 | ECKO Unl (3.4%) | Distinctive: it identifies a product line. |
| `seller` | 6.0% | 534 | RetailNet (5.0%) | Distinctive, but it describes the vendor, not the product. |
| `product_details` | 0.0% | 95.3% of the products have a different text | – | The richest field: fit, sleeve, fabric, color, gender, neck. |

Source: `field_selection_results/field_profile.csv` (columns `empty_pct`, `distinct_raw_values` and `most_common_pct`).

### 2.4 How the choice was evaluated

The 40 products of `validation_labels.csv` (2 queries) were ranked with BM25 for every combination of fields (127 combinations, all merged into one text). A combination is better when the relevant products (label 1) are ranked above the non-relevant ones (label 0). This is measured with the AUC: 1.0 means every 1 is above every 0, and 0.5 is random.

| Field | AUC alone | Change in AUC when it is removed from the full set |
|---|---|---|
| `title` | 0.600 | −0.076 |
| `description` | 0.586 | −0.064 |
| `product_details` | 0.868 | −0.035 |
| `category`, `sub_category`, `brand`, `seller` | 0.500 | 0.000 |

Source: `field_selection_results/ablation.csv` for this table, `combinations.csv` for the scores of all combinations, and `rankings.csv` for the position of every labeled product. The full set scores 0.862, the best combination (`product_details` alone) 0.868, and the chosen combination 0.862. The difference between them is within the noise of a sample this small.

The evaluation was run again after joining `t-shirt` (section 1.1), and **all the scores were identical**, so the decision did not change. This is expected: neither query contains the word "shirt" (query 1 uses `sweatshirt`, which is a different term), so the labeled products cannot show the effect of that step.

### 2.5 Decision for each field

- **`title` (included).** It is weak alone because the labeled products already match the query words in the title, but removing it hurts the most (−0.076). It is also expected to be what finds the candidates in the whole collection, although the labeled products cannot test this.
- **`description` (included).** Removing it costs −0.064. It is empty for 39.7% of the products, so it cannot be the only source of terms.
- **`product_details` (included).** It is the strongest field: alone it reaches 0.868. It carries what separates relevant from non-relevant products. For query 1, `Sleeve = Full Sleeve` is relevant in 13 of 16 products and `Sleeve = Half Sleeve` in 0 of 4. For query 2, `Ideal For = Men` is relevant in 7 of 9 products and `Ideal For = Women` in 3 of 11 (`field_selection_results/field_values.csv`).
- **`sub_category` (included).** It does not change the score for these queries, because no query word appears in it. It is related to relevance (for query 1, `Winter Wear` is relevant in 13 of 16 products and `Topwear` in 0 of 4, see `field_values.csv`), it is short, and it lets queries such as "winter wear" match.
- **`brand` (included).** The evidence cannot confirm it, because neither query mentions a brand and its effect is 0.000. It is included by reasoning: brand names are highly distinctive and users search by brand. Its few tokens are diluted by the longer fields, which is the main weakness of the merged approach.
- **`seller` (not indexed).** Its effect is 0.000, and in the labeled products the same sellers appear on both relevant and non-relevant products (for query 1, `BioworldMerchandising` has 4 relevant products out of 8, see `field_values.csv`). It is distinctive, but it describes the vendor and not the product. A query word could match a seller name by accident.
- **`category` (not indexed).** 96.6% of the products share the same value, so its inverse document frequency is close to zero and it cannot discriminate. Its effect is 0.000.

### 2.6 Limitations of this evidence

- There are only 40 labeled products and 2 queries. Differences of a few thousandths in AUC are noise.
- The labeled products look like the top results of an earlier retrieval, so almost all of them match the query in the title. This understates the value of the title for finding candidates in the whole collection.
- A field that never contains a query word gives the same result whether it is used or not. That is the case for `category`, `sub_category`, `brand` and `seller`, so these four were decided by reasoning and not by the data.
- Some products with identical features have different labels, and some labels look noisy (women's jeans marked relevant for a "men" query). No choice of fields can fix this.

## 3. Numeric fields (`out_of_stock`, `selling_price`, `discount`, `actual_price`, `average_rating`)

### 3.1 Decision

These fields are **not indexed as textual terms**. Each preprocessed product keeps them in a separate `numeric` object, as clean values for the ranking stage:

| Field | Original format | Stored value |
|---|---|---|
| `selling_price` | text, e.g. `"921"` | number (float) |
| `actual_price` | text with thousands separator, e.g. `"2,999"` | number (float) |
| `discount` | text, e.g. `"69% off"` | percentage as a number (`69.0`) |
| `average_rating` | text, e.g. `"3.9"` | number from 1 to 5 |
| `out_of_stock` | boolean | boolean |

The original strings are not changed: they stay in the original record and are the ones returned to the user.

### 3.2 Justification

- **Numbers as text terms lose their meaning.** The tokens `999` and `1000` are unrelated terms, so the index cannot tell that two prices are close, or which one is higher. A text index cannot answer "cheaper" or "better rated" either.
- **They describe the product, not what it is.** Neither of the two reference queries mentions a price, a discount or a rating, so indexing them would add terms that match nothing useful. A term such as `69` could even match an unrelated product by accident.
- **They are natural signals for ranking.** Once converted to numbers they can be used to boost well-rated or discounted products, to push down products that are out of stock, or to filter by price range. This is what the ranking stage needs, and it does not require them in the inverted index.
- **Conversion removes the format problems.** The thousands separator (`"2,999"`) and the text around the discount (`"69% off"`) are removed, so every value of a field has the same type and unit.

### 3.3 Missing values

A missing value is stored as `None` and not filled in. A product without a rating is not a badly rated product, so inventing a value would distort the ranking. The ranking stage has to decide how to treat `None`.

| Field | Missing | Range |
|---|---|---|
| `selling_price` | 2 products | 99 to 7,999 |
| `actual_price` | 777 products (2.8%) | 150 to 12,999 |
| `discount` | 855 products (3.0%) | 1 to 87 (%) |
| `average_rating` | 2,261 products (8.1%) | 1.0 to 5.0 |
| `out_of_stock` | none | 1,644 products (5.9%) are out of stock |

Source: `data_checks_results/numeric_fields.csv`.

### 3.4 Not done yet

The values are only converted, not scaled. If the ranking combines them with the text score, they will need a common scale (for example 0 to 1). The prices are strongly spread (99 to 12,999), so that scaling is a decision for the ranking stage.

## 4. Exploratory data analysis

**TODO.** Word count distribution, average sentence length, vocabulary size, ranking of products by rating, price and discount, top sellers and brands, `out_of_stock` distribution, word clouds, and entity recognition.
