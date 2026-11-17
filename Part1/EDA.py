"""Exploratory data analysis of the fashion products dataset (section 4 of the report).

The tables are saved as CSV files in eda_results/ and the figures as PNG files in eda_results/figures/.
"""
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from wordcloud import WordCloud

from DataPreparation import INDEXED_FIELDS, db_fashion, field_text, normalize_string, preprocess, stop_words

OUTPUT_DIR = Path(__file__).parent / "eda_results"
FIGURES_DIR = OUTPUT_DIR / "figures"
COLOR = "#4C72B0"
QUERIES = {1: "women full sleeve sweatshirt cotton", 2: "men slim jeans blue"}

docs = preprocess(db_fashion)
products = [doc["original"] for doc in docs]


def raw_tokens(product, field):
    """Tokens of a field after normalization only (no stop-word removal, no stemming)."""
    if field == "text":
        return [token for name in INDEXED_FIELDS for token in raw_tokens(product, name)]
    return normalize_string(field_text(product, field)).split()


def save_figure(fig, name):
    """Save a figure as a PNG in the figures folder."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def stats(values):
    """Summary statistics of a list of numbers."""
    series = pd.Series(values, dtype=float)
    return {"n": len(series), "mean": series.mean(), "median": series.median(),
            "p90": series.quantile(0.9), "max": series.max()}


def word_counts():
    """Number of words per product and field, before and after stop-word removal and stemming."""
    rows = []
    for field in ["title", "description", "product_details", "text"]:
        selected = [(product, doc) for product, doc in zip(products, docs) if raw_tokens(product, field)]
        raw = [len(raw_tokens(product, field)) for product, _ in selected]
        processed = [len(doc["processed"][field]) for _, doc in selected]
        rows.append({"field": field, "empty": len(docs) - len(selected), "kind": "raw", **stats(raw)})
        rows.append({"field": field, "empty": len(docs) - len(selected), "kind": "processed", **stats(processed)})
    return pd.DataFrame(rows)


def description_sentences():
    """Words per sentence of every non-empty description. A regex split on .!? is used because many
    descriptions have no space after the period ('look.Designed')."""
    per_description, per_sentence = [], []
    for product in products:
        sentences = [s for s in re.split(r"[.!?]+", product.get("description") or "") if normalize_string(s)]
        if sentences:
            per_description.append(len(sentences))
            per_sentence += [len(normalize_string(s).split()) for s in sentences]
    return per_description, per_sentence


def text_length_figure():
    """One figure with the words per title, words per description and words per sentence."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, field, bins in zip(axes, ["title", "description"], [range(0, 21), range(0, 305, 5)]):
        counts = [len(raw_tokens(product, field)) for product in products if raw_tokens(product, field)]
        ax.hist(counts, bins=bins, color=COLOR)
        ax.set(title=f"Words per {field}", xlabel=f"words (cut at {bins[-1]})", ylabel="products")
    axes[2].hist(description_sentences()[1], bins=range(0, 61), color=COLOR)
    axes[2].set(title="Words per sentence (descriptions)", xlabel="words (cut at 60)", ylabel="sentences")
    save_figure(fig, "text_lengths")


def sentence_lengths():
    """Sentences per description and words per sentence."""
    per_description, per_sentence = description_sentences()
    return pd.DataFrame([{"measure": "sentences_per_description", **stats(per_description)},
                         {"measure": "words_per_sentence", **stats(per_sentence)}])


def vocabulary():
    """Vocabulary size at each preprocessing stage, per field and for the merged indexed text."""
    rows = []
    for field in ["title", "description", "sub_category", "brand", "product_details", "text"]:
        raw = Counter(token for product in products for token in raw_tokens(product, field))
        stemmed = Counter(token for doc in docs for token in doc["processed"][field])
        rows.append({
            "field": field,
            "raw_terms": len(raw),
            "without_stop_words": len([term for term in raw if term not in stop_words]),
            "stemmed_terms": len(stemmed),
            "stemmed_terms_seen_once": sum(count == 1 for count in stemmed.values()),
            "stemmed_tokens": sum(stemmed.values()),
        })

    frequency = Counter(token for doc in docs for token in doc["processed"]["text"])
    document_frequency = Counter(token for doc in docs for token in set(doc["processed"]["text"]))
    top = pd.DataFrame([{"term": term, "frequency": count, "document_frequency": document_frequency[term],
                         "document_frequency_pct": 100 * document_frequency[term] / len(docs)}
                        for term, count in frequency.most_common(25)])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = sorted(frequency.values(), reverse=True)
    ax.loglog(range(1, len(counts) + 1), counts, color=COLOR)
    ax.set(title="Rank-frequency of the indexed terms (log-log)", xlabel="rank", ylabel="frequency")
    save_figure(fig, "zipf")
    return pd.DataFrame(rows), top


def duplicates():
    """How many products share the same title or description."""
    titles = Counter(product["title"] for product in products)
    descriptions = Counter(product["description"] for product in products if product.get("description"))
    summary = pd.DataFrame([
        {"measure": "products", "value": len(products)},
        {"measure": "distinct_titles", "value": len(titles)},
        {"measure": "products_with_repeated_title", "value": sum(c for c in titles.values() if c > 1)},
        {"measure": "non_empty_descriptions", "value": sum(descriptions.values())},
        {"measure": "distinct_descriptions", "value": len(descriptions)},
    ])
    top = pd.DataFrame(titles.most_common(10), columns=["title", "products"])
    top_descriptions = pd.DataFrame([{"description_start": text[:70], "words": len(normalize_string(text).split()),
                                      "products": count} for text, count in descriptions.most_common(10)])
    return summary, top, top_descriptions


def style_code_terms():
    """How many of the product_details terms seen only once come from the 'Style Code' value."""
    frequency, from_style_code = Counter(), set()
    for product in products:
        for detail in product["product_details"]:
            for name, value in detail.items():
                tokens = preprocess([{"pid": "", "title": value}])[0]["processed"]["title"]
                frequency.update(tokens)
                if name == "Style Code":
                    from_style_code.update(tokens)
    seen_once = {term for term, count in frequency.items() if count == 1}
    return pd.DataFrame([
        {"measure": "products_with_style_code",
         "value": sum(any("Style Code" in detail for detail in product["product_details"]) for product in products)},
        {"measure": "product_details_terms_seen_once", "value": len(seen_once)},
        {"measure": "of_which_from_style_code", "value": len(seen_once & from_style_code)},
    ])


def numeric_values(field):
    """Values of a numeric field, without the missing ones."""
    return [doc["numeric"][field] for doc in docs if doc["numeric"][field] is not None]


def numeric_distributions():
    """Distribution of price, discount and rating."""
    fields = ["selling_price", "actual_price", "discount", "average_rating"]
    rows = []
    for field in fields:
        series = pd.Series(numeric_values(field))
        rows.append({"field": field, "n": len(series), "mean": series.mean(), "q1": series.quantile(0.25),
                     "median": series.median(), "q3": series.quantile(0.75)})

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    bins = [range(0, 4100, 100), range(0, 92, 2), [1 + step / 10 - 0.05 for step in range(42)]]
    labels = ["selling price (cut at 4,000)", "discount (%)", "average rating"]
    for ax, field, field_bins, label in zip(axes, ["selling_price", "discount", "average_rating"], bins, labels):
        ax.hist(numeric_values(field), bins=field_bins, color=COLOR)
        ax.set(title=field, xlabel=label, ylabel="products")
    save_figure(fig, "numeric_hist")
    return pd.DataFrame(rows)


def ranking(field, ascending=False, n=10):
    """Top n products by a numeric field; ties broken by pid so the result is reproducible."""
    rows = [{"pid": doc["pid"], "title": doc["original"]["title"], "brand": doc["original"]["brand"],
             field: doc["numeric"][field]} for doc in docs if doc["numeric"][field] is not None]
    table = pd.DataFrame(rows).sort_values([field, "pid"], ascending=[ascending, True])
    return table.head(n)


def rankings():
    """Products ranked by rating, price and discount."""
    top_rating = ranking("average_rating")
    top_rating["products_with_this_rating"] = sum(value == 5.0 for value in numeric_values("average_rating"))
    prices = pd.concat([ranking("selling_price").assign(rank="most_expensive"),
                        ranking("selling_price", ascending=True).assign(rank="cheapest")])
    return top_rating, prices, ranking("discount")


def group_table(key, n=15):
    """Products, share, out-of-stock percentage and mean rating of the most common values of a field."""
    rows = []
    for value, count in Counter(product[key] or "(empty)" for product in products).most_common(n):
        group = [doc for doc in docs if (doc["original"][key] or "(empty)") == value]
        ratings = [doc["numeric"]["average_rating"] for doc in group if doc["numeric"]["average_rating"]]
        rows.append({
            key: value,
            "products": count,
            "share_pct": 100 * count / len(docs),
            "out_of_stock_pct": 100 * sum(doc["numeric"]["out_of_stock"] for doc in group) / count,
            "mean_rating": sum(ratings) / len(ratings) if ratings else None,
        })
    return pd.DataFrame(rows)


def top_sellers_brands():
    """Sellers and brands with the most products."""
    sellers, brands = group_table("seller"), group_table("brand")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, table, key in zip(axes, [brands, sellers], ["brand", "seller"]):
        ax.barh(table[key][::-1], table["products"][::-1], color=COLOR)
        ax.set(title=f"Top 15 {key}s", xlabel="products")
    save_figure(fig, "top_brands_sellers")
    return sellers, brands


def stock():
    """Out-of-stock products, overall and by sub_category (only those with at least 100 products)."""
    rows = [{"sub_category": "ALL", "products": len(docs),
             "out_of_stock": sum(doc["numeric"]["out_of_stock"] for doc in docs)}]
    for value, count in Counter(product["sub_category"] for product in products).most_common():
        if count >= 100:
            out = sum(doc["numeric"]["out_of_stock"] for doc in docs if doc["original"]["sub_category"] == value)
            rows.append({"sub_category": value, "products": count, "out_of_stock": out})
    table = pd.DataFrame(rows)
    table["out_of_stock_pct"] = 100 * table["out_of_stock"] / table["products"]

    fig, ax = plt.subplots(figsize=(7, 4))
    shown = table[table["sub_category"] != "ALL"].sort_values("out_of_stock_pct")
    ax.barh(shown["sub_category"], shown["out_of_stock_pct"], color=COLOR)
    ax.axvline(table["out_of_stock_pct"][0], color="grey", linestyle="--", label="all products")
    ax.set(title="Out-of-stock products by sub_category", xlabel="% out of stock")
    ax.legend()
    save_figure(fig, "out_of_stock")
    return table


def stock_vs_seller():
    """Out-of-stock products against products with an empty seller."""
    counts = Counter((bool(product["seller"]), bool(product["out_of_stock"])) for product in products)
    return pd.DataFrame([{"seller_present": seller, "out_of_stock": out, "products": count}
                         for (seller, out), count in sorted(counts.items())])


def brand_truncation():
    """Compare the brand with the first part of the URL path, which holds the full brand name
    ('York' against 'yorker-solid-men-...')."""
    rows, examples = Counter(), Counter()
    for product in products:
        brand = "-".join(normalize_string(product["brand"]).split())
        slug = product["url"].split("/")[3]
        if not brand:
            rows["empty_brand"] += 1
        elif slug.startswith(brand + "-"):
            rows["brand_equals_url_name"] += 1
        elif slug.startswith(brand):
            rows["brand_truncated"] += 1
            examples[(product["brand"], "-".join(slug.split("-")[:brand.count("-") + 1]))] += 1
        else:
            rows["brand_not_in_url"] += 1
    summary = pd.DataFrame([{"case": case, "products": count, "products_pct": 100 * count / len(products)}
                            for case, count in rows.most_common()])
    top = pd.DataFrame([{"brand": brand, "url_start": start, "products": count}
                        for (brand, start), count in examples.most_common(10)])
    return summary, top


def sub_categories():
    """Number of products in each sub_category."""
    counts = Counter(product["sub_category"] for product in products)
    return pd.DataFrame(counts.most_common(), columns=["sub_category", "products"])


def word_clouds():
    """Word clouds of the title and description words, without stop words and not stemmed (stems such as
    'typographi' are hard to read). Titles and descriptions side by side in one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    for ax, field in zip(axes, ["title", "description"]):
        words = Counter(token for product in products for token in raw_tokens(product, field)
                        if token not in stop_words)
        cloud = WordCloud(width=1000, height=550, background_color="white", random_state=0)
        ax.imshow(cloud.generate_from_frequencies(words), interpolation="bilinear")
        ax.set_title(f"{field.capitalize()}s")
        ax.axis("off")
    save_figure(fig, "wordclouds")


def entities():
    """Named entities found by spaCy (en_core_web_sm) in the non-empty descriptions."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    texts = [product["description"] for product in products if product.get("description")]
    start = time.time()
    labels, mentions, examples = Counter(), Counter(), {}
    docs_with_label = Counter()
    for parsed in nlp.pipe(texts, batch_size=256):
        for entity in parsed.ents:
            labels[entity.label_] += 1
            mentions[(entity.text, entity.label_)] += 1
        docs_with_label.update({entity.label_ for entity in parsed.ents})
    print(f"Entity recognition on {len(texts)} descriptions: {time.time() - start:.0f} s")

    for (text, label), _ in mentions.most_common():
        examples.setdefault(label, [])
        if len(examples[label]) < 5:
            examples[label].append(text)
    by_label = pd.DataFrame([{"label": label, "mentions": count, "descriptions": docs_with_label[label],
                              "descriptions_pct": 100 * docs_with_label[label] / len(texts),
                              "top_examples": " | ".join(examples[label])}
                             for label, count in labels.most_common()])
    top = pd.DataFrame([{"entity": text, "label": label, "mentions": count}
                        for (text, label), count in mentions.most_common(30)])
    return by_label, top


def query_term_df():
    """Share of products whose indexed text contains each word of the two reference queries."""
    rows = []
    for query_id, query in QUERIES.items():
        for word in query.split():
            term = preprocess_word(word)
            matches = sum(term in doc["processed"]["text"] for doc in docs)
            rows.append({"query_id": query_id, "word": word, "term": term, "products": matches,
                         "products_pct": 100 * matches / len(docs)})
    return pd.DataFrame(rows)


def preprocess_word(word):
    """Stemmed form of a query word, the same way the products are processed."""
    return preprocess([{"pid": "", "title": word}])[0]["processed"]["title"][0]


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
vocabulary_table, top_terms = vocabulary()
duplicates_summary, repeated_titles, repeated_descriptions = duplicates()
top_rating, prices, top_discount = rankings()
top_sellers, top_brands = top_sellers_brands()
entities_by_label, top_entities = entities()
brand_cases, truncated_brands = brand_truncation()
word_clouds()
text_length_figure()
tables = {
    "word_counts": word_counts(),
    "sentence_lengths": sentence_lengths(),
    "vocabulary": vocabulary_table,
    "top_terms": top_terms,
    "duplicates": duplicates_summary,
    "repeated_titles": repeated_titles,
    "repeated_descriptions": repeated_descriptions,
    "style_code_terms": style_code_terms(),
    "numeric_summary": numeric_distributions(),
    "top_rating": top_rating,
    "prices": prices,
    "top_discount": top_discount,
    "top_sellers": top_sellers,
    "top_brands": top_brands,
    "out_of_stock": stock(),
    "stock_vs_seller": stock_vs_seller(),
    "brand_truncation": brand_cases,
    "truncated_brands": truncated_brands,
    "sub_categories": sub_categories(),
    "entities_by_label": entities_by_label,
    "top_entities": top_entities,
    "query_term_df": query_term_df(),
}
for name, table in tables.items():
    table.round(3).to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

print(f"Results saved in {OUTPUT_DIR}: {', '.join(f'{name}.csv' for name in tables)}")
