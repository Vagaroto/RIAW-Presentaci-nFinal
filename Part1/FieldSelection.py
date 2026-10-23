"""Evidence to decide which fields to index (used or not, no weights).

For every subset of fields, the chosen fields are merged into one text and the labeled products
are ranked with BM25 (idf and average length from the whole corpus). A subset is good when the
relevant products (1) end up at the top and the non-relevant ones (0) at the bottom.
The results are saved as CSV files in field_selection_results/ to be referenced in the report.
"""
import itertools
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import pandas as pd

from DataPreparation import FIELDS, INDEXED_FIELDS, db_fashion, field_text, preprocess, preprocess_text

QUERIES = {1: "women full sleeve sweatshirt cotton", 2: "men slim jeans blue"}
K1, B = 1.2, 0.75
OUTPUT_DIR = Path(__file__).parent / "field_selection_results"

validation = pd.read_csv(Path(__file__).parent.parent / "validation_labels.csv")
products = {product["pid"]: product for product in db_fashion}
docs = {doc["pid"]: doc["processed"] for doc in preprocess(db_fashion)}
query_terms = {q: set(preprocess_text(text)) for q, text in QUERIES.items()}
titles = dict(zip(validation.pid, validation.title))


# Ranking and evaluation

def merged_tokens(pid, fields):
    return [token for field in fields for token in docs[pid][field]]


def rank(fields):
    """Labeled products of each query, best score first, as {query_id: [(score, label, pid), ...]}."""
    corpus = {pid: merged_tokens(pid, fields) for pid in docs}
    n = len(corpus)
    avg_len = sum(len(tokens) for tokens in corpus.values()) / n
    rankings = {}
    for query_id, terms in query_terms.items():
        df = Counter(token for tokens in corpus.values() for token in set(tokens) & terms)
        rows = validation[validation.query_id == query_id]
        scored = []
        for pid, label in zip(rows.pid, rows.labels):
            tokens = corpus[pid]
            tf = Counter(tokens)
            score = sum(
                math.log(n / df[t]) * (K1 + 1) * tf[t] / (K1 * ((1 - B) + B * len(tokens) / avg_len) + tf[t])
                for t in terms if df[t]
            )
            scored.append((score, label, pid))
        rankings[query_id] = sorted(scored, reverse=True)
    return rankings


def auc(ranking):
    """Probability that a relevant product is ranked above a non-relevant one (ties count half)."""
    relevant = [score for score, label, _ in ranking if label == 1]
    not_relevant = [score for score, label, _ in ranking if label == 0]
    wins = sum((r > n) + 0.5 * (r == n) for r in relevant for n in not_relevant)
    return wins / (len(relevant) * len(not_relevant))


# evaluations[fields] = (mean AUC, [AUC of each query], rankings)
evaluations = {}
for size in range(1, len(FIELDS) + 1):
    for fields in itertools.combinations(FIELDS, size):
        rankings = rank(fields)
        aucs = [auc(ranking) for ranking in rankings.values()]
        evaluations[fields] = (mean(aucs), aucs, rankings)

# stable sort: among equal scores the combination with fewer fields comes first
ordered = sorted(evaluations, key=lambda fields: -evaluations[fields][0])
ALL_FIELDS = tuple(FIELDS)
CHOSEN = tuple(INDEXED_FIELDS)  # the combination used in the project


# Result tables

def field_profile():
    """How filled and how distinctive each field is over the whole dataset."""
    rows = []
    for field in FIELDS:
        values = [" ".join(doc[field]) for doc in docs.values()]
        filled = [value for value in values if value]
        top_value, top_count = Counter(filled).most_common(1)[0]
        rows.append({
            "field": field,
            "empty_pct": 100 * (1 - len(filled) / len(values)),
            "avg_tokens": mean(len(doc[field]) for doc in docs.values()),
            "distinct_raw_values": len({field_text(product, field) for product in db_fashion} - {""}),
            "distinct_values_pct": 100 * len(set(filled)) / len(values),
            "most_common_value": top_value,
            "most_common_pct": 100 * top_count / len(values),
        })
    return pd.DataFrame(rows)


def query_term_matches():
    """Average number of query terms found in each field, for relevant and non-relevant products."""
    rows = []
    for query_id, terms in query_terms.items():
        group = validation[validation.query_id == query_id]
        for field in FIELDS:
            row = {"query": query_id, "field": field}
            for name, label in (("relevant", 1), ("not_relevant", 0)):
                pids = group.pid[group.labels == label]
                row[f"avg_terms_{name}"] = mean(len(terms & set(docs[pid][field])) for pid in pids)
            rows.append(row)
    return pd.DataFrame(rows)


def product_values(product):
    values = [("sub_category", product.get("sub_category")), ("brand", product.get("brand")),
              ("seller", product.get("seller"))]
    for detail in product.get("product_details") or []:
        values += [("product_details", f"{key} = {value}") for key, value in detail.items()]
    return [(field, str(value)) for field, value in values if value]


def field_values():
    """Field values in at least 4 labeled products. For product_details only those that are at least 70%
    or at most 30% relevant (it has too many values); for the other fields all of them."""
    rows = []
    for query_id in QUERIES:
        group = validation[validation.query_id == query_id]
        total, relevant = Counter(), Counter()
        for pid, label in zip(group.pid, group.labels):
            for entry in set(product_values(products[pid])):
                total[entry] += 1
                relevant[entry] += label
        for (field, value), count in total.items():
            share = relevant[(field, value)] / count
            if count >= 4 and (field != "product_details" or not 0.3 < share < 0.7):
                rows.append({"query": query_id, "field": field, "value": value,
                             "relevant": relevant[(field, value)], "total": count, "pct_relevant": 100 * share})
    return pd.DataFrame(rows).sort_values(["query", "field", "total"], ascending=[True, True, False])


def ablation():
    """AUC of each field alone and the change when it is removed from the full set."""
    full_auc = evaluations[ALL_FIELDS][0]
    rows = []
    for field in FIELDS:
        without = tuple(f for f in FIELDS if f != field)
        rows.append({"field": field, "auc_alone": evaluations[(field,)][0],
                     "auc_without_it": evaluations[without][0],
                     "change": evaluations[without][0] - full_auc})
    return pd.DataFrame(rows)


def combinations():
    """All field combinations, best first."""
    return pd.DataFrame([
        {"mean_auc": evaluations[fields][0],
         **{f"auc_q{q}": auc_q for q, auc_q in zip(QUERIES, evaluations[fields][1])},
         "n_fields": len(fields), "fields": " + ".join(fields),
         "note": "chosen" if fields == CHOSEN else "all fields" if fields == ALL_FIELDS else ""}
        for fields in ordered
    ])


def rankings_table():
    """Position, label and score of every labeled product for the best, chosen and title-only combinations."""
    rows = []
    for name, fields in [("best", ordered[0]), ("chosen", CHOSEN), ("title_only", ("title",))]:
        for query_id, ranking in evaluations[fields][2].items():
            for position, (score, label, pid) in enumerate(ranking, start=1):
                rows.append({"combination": name, "fields": " + ".join(fields), "query": query_id,
                             "position": position, "label": label, "score": score, "title": titles[pid]})
    return pd.DataFrame(rows)


OUTPUT_DIR.mkdir(exist_ok=True)
tables = {
    "field_profile": field_profile(),
    "query_term_matches": query_term_matches(),
    "field_values": field_values(),
    "ablation": ablation(),
    "combinations": combinations(),
    "rankings": rankings_table(),
}
for name, table in tables.items():
    table.round(3).to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

print(f"Results saved in {OUTPUT_DIR}: {', '.join(f'{name}.csv' for name in tables)}")
for fields in ordered[:5] + [CHOSEN]:
    print(f"  {evaluations[fields][0]:.3f}  {' + '.join(fields)}")
