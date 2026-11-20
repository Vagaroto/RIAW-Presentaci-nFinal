"""Clean base for Part 3: index, AND search and evaluation, with one shape for every ranking.

From Part 2 we import only what is clean (queries, labels, metrics). The index, the search and the
evaluation are rewritten here, so Part 3 does not depend on the rest of Part 2.

Every ranking function has the same signature:
    ranking(query_terms, candidates, prep) -> [(pid, score)], best first
so they can be swapped in search() and evaluate().
"""
import math
from collections import Counter, defaultdict

from part1.DataPreparation import db_fashion, preprocess, preprocess_text
from part2.InvertedIndex import TEST_QUERIES
from part2.Evaluation import (VALIDATION_QUERIES, VALIDATION_LABELS, TEST_LABELS,
                              read_labels, evaluate_query)


# ---------- computed once ----------

def build_index(docs: list) -> dict:
    """index[term] = {"idf": ln(N / df), "postings": {pid: tf}}; the postings keep the order of docs."""
    postings = defaultdict(dict)
    for doc in docs:
        for term, tf in Counter(doc["processed"]["text"]).items():
            postings[term][doc["pid"]] = tf
    N = len(docs)
    return {term: {"idf": math.log(N / len(p)), "postings": p} for term, p in postings.items()}

def tfidf(tf: float, idf: float) -> float:
    """Weight of a term in a text: raw tf times idf (the same formula as Part 2)."""
    return tf * idf

def doc_norms(index: dict) -> dict:
    """pid -> length (L2 norm) of the doc's tf-idf vector, over all its terms."""
    squares = defaultdict(float)
    for entry in index.values():
        for pid, tf in entry["postings"].items():
            squares[pid] += tfidf(tf, entry["idf"]) ** 2
    return {pid: math.sqrt(s) for pid, s in squares.items()}

def prepare(docs: list) -> dict:
    """Everything the rankings need, computed once."""
    index = build_index(docs)
    lengths = {doc["pid"]: len(doc["processed"]["text"]) for doc in docs}
    return {
        "index": index,
        "norms": doc_norms(index),
        "lengths": lengths,                                   # number of tokens of each doc
        "avg_length": sum(lengths.values()) / len(lengths),
        "numeric": {doc["pid"]: doc["numeric"] for doc in docs},
        "original": {doc["pid"]: doc["original"] for doc in docs},   # to show results
        "tokens": {doc["pid"]: doc["processed"]["text"] for doc in docs},   # for word2vec
    }


# ---------- per query ----------

def matching(query_terms: list, index: dict) -> list:
    """Pids of the docs that contain ALL query terms (AND), in the order of docs. Empty if any term is unknown."""
    if not query_terms or any(t not in index for t in query_terms):
        return []
    first, *rest = query_terms
    return [pid for pid in index[first]["postings"]
            if all(pid in index[t]["postings"] for t in rest)]

def rank(scores: dict) -> list:
    """[(pid, score)] sorted best first (ties keep their order)."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def search(query: str, prep: dict, ranking) -> list:
    """Clean the query, keep the AND matches, sort them with the given ranking."""
    query_terms = preprocess_text(query)
    candidates = matching(query_terms, prep["index"])
    return ranking(query_terms, candidates, prep)


# ---------- reference ranking ----------

def tfidf_sum(query_terms: list, candidates: list, prep: dict) -> list:
    """Part 2's ranking: sum over query terms of tf * idf, no normalization. Kept as the baseline."""
    index = prep["index"]
    scores = {pid: 0.0 for pid in candidates}
    for term in query_terms:
        entry = index[term]
        for pid in candidates:
            scores[pid] += tfidf(entry["postings"][pid], entry["idf"])
    return rank(scores)


# ---------- evaluation ----------

def all_queries() -> dict:
    """{query_id: query}: the 2 validation queries and the 5 test queries of Part 2."""
    return {**VALIDATION_QUERIES, **dict(enumerate(TEST_QUERIES, start=3))}

def all_labels() -> dict:
    """{query_id: set of relevant pids} for the 7 queries."""
    return {**read_labels(VALIDATION_LABELS), **read_labels(TEST_LABELS)}

def evaluate(prep: dict, ranking) -> list:
    """One row of metrics per query, plus a MEAN row (AP -> MAP, RR -> MRR)."""
    labels = all_labels()
    rows = []
    for query_id, query in all_queries().items():
        ranked = [pid for pid, _ in search(query, prep, ranking)]
        rows.append({"query_id": query_id, "query": query, **evaluate_query(ranked, labels[query_id])})
    names = [n for n in rows[0] if n not in ("query_id", "query")]
    rows.append({"query_id": "MEAN", "query": "", **{n: sum(r[n] for r in rows) / len(rows) for n in names}})
    return rows


if __name__ == "__main__":
    prep = prepare(preprocess(db_fashion))
    for row in evaluate(prep, tfidf_sum):
        print(row["query_id"], f"P@10={row['P@10']:.3f} AP={row['AP']:.3f} RR={row['RR']:.3f}")
