import json
import re
import unicodedata

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


with open('fashion_products_dataset.json') as f:
    db_fashion = json.load(f)

df = pd.json_normalize(db_fashion)  # o pd.DataFrame(db_fashion)
print(df.shape)
print(df.head(5))
print(df.tail(5))


# 1º Part: DATA PREPARATION

# Text normalization

def normalize_string(text):
    """Unicode-normalize, strip diacritics, lowercase, join t-shirt, remove punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"\bt[\s-]+shirt", "tshirt", text)  # otherwise the stop word "t" turns t-shirt into shirt
    return " ".join(re.findall(r"\w+", text))


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def preprocess_text(text):
    """Normalize, remove stop words and stem."""
    return [stemmer.stem(t) for t in normalize_string(text).split() if t not in stop_words]


# Document preprocessing

# Candidate fields, all of them are evaluated in FieldSelection.py
FIELDS = ["title", "description", "category", "sub_category", "brand", "product_details", "seller"]

# Decision (evidence in field_selection_results/, justification in Report_Part1.md):
# the fields below are merged into one text that is indexed, without weights.
# Not indexed: category (almost all products share the same value) and seller (distinctive
# but not relevant when searching). They stay in the original product so they can be returned.
INDEXED_FIELDS = ["title", "description", "sub_category", "brand", "product_details"]
print(FIELDS)

def field_text(product, field):
    """Raw text of a field; product_details is a list of {name: value} dicts, only the values are kept."""
    if field == "product_details":
        return " ".join(str(value) for detail in product.get(field) or [] for value in detail.values())
    return product.get(field) or ""


def to_number(text):
    """First number in a string such as '2,999' or '69% off'; None when the field is empty."""
    match = re.search(r"\d[\d,]*\.?\d*", str(text or ""))
    return float(match.group().replace(",", "")) if match else None


def numeric_values(product):
    """Values kept for ranking, not indexed: prices, discount in percent, rating (1 to 5) and stock flag."""
    return {
        "selling_price": to_number(product.get("selling_price")),
        "actual_price": to_number(product.get("actual_price")),
        "discount": to_number(product.get("discount")),
        "average_rating": to_number(product.get("average_rating")),
        "out_of_stock": bool(product.get("out_of_stock")),
    }


def preprocess(db_fashion):
    """Pair each untouched product with its processed tokens (one list per field and the merged text to
    index) and its numeric values (for ranking)."""
    docs = []
    for product in db_fashion:
        processed = {field: preprocess_text(field_text(product, field)) for field in FIELDS}
        processed["text"] = [token for field in INDEXED_FIELDS for token in processed[field]]
        docs.append({
            "pid": product["pid"],
            "original": product,
            "processed": processed,
            "numeric": numeric_values(product),
        })
    return docs
