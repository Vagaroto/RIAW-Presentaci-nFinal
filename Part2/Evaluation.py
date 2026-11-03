import csv
import math
import os

from part1.DataPreparation import db_fashion, preprocess
from part2.InvertedIndex import build_index, search, TEST_QUERIES

K_VALUES = [5, 10, 20]
FOLDER = os.path.dirname(os.path.abspath(__file__))
VALIDATION_LABELS = os.path.join(FOLDER, "..", "validation_labels.csv")
TEST_LABELS = os.path.join(FOLDER, "test_labels.csv")          # filled in by hand (expert judges)
RESULTS = os.path.join(FOLDER, "evaluation_results.csv")

VALIDATION_QUERIES = {1: "women full sleeve sweatshirt cotton", 2: "men slim jeans blue"}


# ---------- metrics: ranked = list of pids (best first), relevant = set of relevant pids ----------

def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    """Relevant products in the top k, divided by k."""
    return sum(pid in relevant for pid in ranked[:k]) / k

def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    """Relevant products in the top k, divided by all the relevant products."""
    return sum(pid in relevant for pid in ranked[:k]) / len(relevant)

def f1_at_k(ranked: list, relevant: set, k: int) -> float:
    """Harmonic mean of P@k and R@k (0 if both are 0)."""
    p, r = precision_at_k(ranked, relevant, k), recall_at_k(ranked, relevant, k)
    return 2 * p * r / (p + r) if p + r else 0.0

def average_precision_at_k(ranked: list, relevant: set, k: int) -> float:
    """Mean of P@i over the ranks i <= k holding a relevant product, divided by min(k, #relevant)."""
    hits, total = 0, 0.0
    for i, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            hits += 1
            total += hits / i
    return total / min(k, len(relevant))

def reciprocal_rank(ranked: list, relevant: set) -> float:
    """1 / rank of the first relevant product (0 if there is none in the ranking)."""
    for i, pid in enumerate(ranked, start=1):
        if pid in relevant:
            return 1 / i
    return 0.0

def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    """Binary-gain NDCG: DCG of the ranking divided by the DCG of an ideal ranking."""
    dcg = sum(1 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], start=1) if pid in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


# ---------- running them ----------

def evaluate_query(ranked: list, relevant: set) -> dict:
    """All the metrics of one query. MAP and MRR are the means of AP and RR over queries (see evaluate)."""
    scores = {}
    for k in K_VALUES:
        scores[f"P@{k}"] = precision_at_k(ranked, relevant, k)
        scores[f"R@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"AP@{k}"] = average_precision_at_k(ranked, relevant, k)
        scores[f"F1@{k}"] = f1_at_k(ranked, relevant, k)
        scores[f"NDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
    # AP over the whole ranking (divided by the number of relevant products); its mean over queries is the MAP
    scores["AP"] = average_precision_at_k(ranked, relevant, max(len(ranked), len(relevant)))
    scores["RR"] = reciprocal_rank(ranked, relevant)
    return scores

def evaluate(inverted_index: dict, queries: dict, relevant_by_query: dict) -> list:
    """One row per query plus a final MEAN row (its AP@k columns are the MAP@k, AP is the MAP, RR is the MRR)."""
    rows = []
    for query_id, query in queries.items():
        ranked = [pid for pid, _ in search(inverted_index, query)]
        rows.append({"query_id": query_id, "query": query,
                     **evaluate_query(ranked, relevant_by_query[query_id])})
    metric_names = [name for name in rows[0] if name not in ("query_id", "query")]
    rows.append({"query_id": "MEAN", "query": "",
                 **{name: sum(row[name] for row in rows) / len(rows) for name in metric_names}})
    return rows


# ---------- labels ----------

def read_labels(path: str) -> dict:
    """{query_id: set of relevant pids} from a CSV with the columns query_id, pid, labels."""
    relevant = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            relevant.setdefault(int(row["query_id"]), set())
            if row["labels"].strip() == "1":
                relevant[int(row["query_id"])].add(row["pid"])
    return relevant

def export_to_label(inverted_index: dict, docs: list, top: int = 20):
    """Write the top results of each test query to test_labels.csv, to be judged by hand (never overwrites)."""
    if os.path.exists(TEST_LABELS):
        return
    originals = {doc["pid"]: doc["original"] for doc in docs}
    with open(TEST_LABELS, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "pid", "title", "brand", "description", "labels"])
        for query_id, query in enumerate(TEST_QUERIES, start=3):
            for pid, _ in search(inverted_index, query)[:top]:
                original = originals[pid]
                writer.writerow([query_id, query, pid, original["title"], original.get("brand"),
                                 (original.get("description") or "")[:200], ""])

def labels_complete(path: str) -> bool:
    """True when every row of the hand-made labels file has a 0 or a 1."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return all(row["labels"].strip() in ("0", "1") for row in csv.DictReader(f))

def save(rows: list):
    """Write the rows to evaluation_results.csv, numbers rounded to 3 decimals."""
    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 3) if isinstance(value, float) else value
                             for key, value in row.items()})


if __name__ == "__main__":
    docs = preprocess(db_fashion)
    inverted_index = build_index(docs)

    # Part 2, items 1 and 2: the two validation queries
    rows = evaluate(inverted_index, VALIDATION_QUERIES, read_labels(VALIDATION_LABELS))

    # Part 2, item 3a: our own queries, once the labels have been filled in by hand
    export_to_label(inverted_index, docs)
    if labels_complete(TEST_LABELS):
        test_queries = dict(enumerate(TEST_QUERIES, start=3))
        rows += evaluate(inverted_index, test_queries, read_labels(TEST_LABELS))
    else:
        print(f"\n{TEST_LABELS} still has empty labels: fill the 'labels' column (1/0) and run again.")

    save(rows)
    print()
    for row in rows:
        print(row["query_id"], " ".join(f"{name}={value:.3f}" for name, value in row.items()
                                       if isinstance(value, float)))
