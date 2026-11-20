"""Part 3: ranking & filtering. The rankings live here; the index and the search come from Base.

Assignment (part3/IRWA_project_part_3.pdf). Pipeline: query -> docs containing ALL its terms (AND) -> ranked.
1. Three rankings:
   a. TF-IDF + cosine similarity                                 -> tf_idf_cosine
   b. BM25                                                       -> bm25
   c. Our own score: TF-IDF cosine x rating x stock              -> tikitikiscore
   Report: how TF-IDF and BM25 rank differently, pros and cons of each; justify our score (pros and cons).
2. word2vec + cosine: a text is the average of its word vectors  -> word2vec_cosine
   Top 20 for each of the 5 Part 2 queries                       -> results/word2vec_top20.csv
3. Report only: a better representation than word2vec? (doc2vec, sentence2vec: pros and cons)

Every ranking: ranking(query_terms, candidates, prep) -> [(pid, score)], best first.
prep["index"][t]["idf"], prep["index"][t]["postings"][pid] (tf), prep["norms"][pid],
prep["lengths"][pid], prep["avg_length"], prep["numeric"][pid], prep["original"][pid],
prep["tokens"][pid], prep["w2v"] (set in __main__).

Run from the repo root: py -m part3.Scoring. Writes, in part3/results/:
  evaluation.csv       metrics of every ranking on the 7 labeled queries (one row per ranking and query)
  rankings_top10.csv   top 10 of every ranking for every query, to compare the orderings
  word2vec_top20.csv   item 2
"""
from part1.DataPreparation import db_fashion, preprocess
from part2.InvertedIndex import TEST_QUERIES
from part3.Base import prepare, search, rank, tfidf, tfidf_sum, evaluate, all_queries
from pathlib import Path
import csv

import numpy as np
from gensim.models import Word2Vec


K1 = 1.4
B = 0.25

def tf_idf_cosine(query_terms: list, candidates: list, prep: dict) -> list:
    """Rank candidate docs by tf-idf cosine with the query (query norm dropped: same for every doc)."""
    index, norms = prep["index"], prep["norms"]
    score = {doc: 0.0 for doc in candidates}

    for term in set(query_terms):
        idf = index[term]["idf"]
        weight_q = tfidf(query_terms.count(term), idf)
        for doc in candidates:
            weight_d = tfidf(index[term]["postings"][doc], idf)
            score[doc] = score[doc] + weight_q * weight_d

    for doc in candidates:
        score[doc] = score[doc] / norms[doc]

    return rank(score)


def bm25(query_terms: list, candidates: list, prep: dict) -> list:
    """Rank candidate docs by BM25: tf saturates (K1) and long docs are penalized (B)."""
    
    index, lengths, l_ave = prep["index"], prep["lengths"], prep["avg_length"]
    score = {doc: 0.0 for doc in candidates}
    
    for term in query_terms:
        idf = index[term]["idf"]
        for doc in candidates:
            tf = index[term]["postings"][doc]
            score[doc] += idf * (K1 + 1) * tf / (K1 * ((1 - B) + B * lengths[doc] / l_ave) + tf)
    
    return rank(score)


NO_RATING = 0.7      # rating factor of a product without rating: unknown, not bad
OUT_OF_STOCK = 0.5   # stock factor of an out-of-stock product: demoted, not removed

def tikitikiscore(query_terms: list, candidates: list, prep: dict) -> list:
    """TF-IDF cosine x rating / 5 (NO_RATING if none) x OUT_OF_STOCK when it cannot be bought."""
    score = dict(tf_idf_cosine(query_terms, candidates, prep))
    for doc in candidates:
        numeric = prep["numeric"][doc]
        rating = numeric["average_rating"]
        rating_factor = rating / 5 if rating is not None else NO_RATING
        stock_factor = OUT_OF_STOCK if numeric["out_of_stock"] else 1.0
        score[doc] = score[doc] * rating_factor * stock_factor
    return rank(score)


def train_word2vec(prep: dict) -> Word2Vec:
    """One model for the whole corpus: each doc's token list is a sentence."""
    return Word2Vec(sentences=list(prep["tokens"].values()), vector_size=100, window=5, min_count=1,
                    seed=0, workers=1)   # one worker: same vectors on every run

def text_vector(tokens: list, model: Word2Vec) -> np.ndarray:
    """Average of the word vectors of the text (words unknown to the model are skipped)."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def cosine(a, b) -> float:
    """Cosine similarity, 0 if a vector is null."""
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0

def word2vec_cosine(query_terms: list, candidates: list, prep: dict) -> list:
    """Rank candidates by cosine between the averaged query vector and the averaged doc vector."""
    model = prep["w2v"]
    q = text_vector(query_terms, model)
    return rank({doc: cosine(q, text_vector(prep["tokens"][doc], model)) for doc in candidates})


RANKINGS = {
    "tfidf_sum (Part 2)": tfidf_sum,
    "tf_idf_cosine": tf_idf_cosine,
    "bm25": bm25,
    "tikitikiscore": tikitikiscore,
    "word2vec_cosine": word2vec_cosine,
}
RESULTS = Path(__file__).parent / "results"


def top_k(prep: dict, ranking, queries: dict, k: int) -> list:
    """One row per result: the top k of the ranking for each query."""
    rows = []
    for query_id, query in queries.items():
        for position, (pid, score) in enumerate(search(query, prep, ranking)[:k], start=1):
            rows.append({"query_id": query_id, "query": query, "rank": position, "pid": pid,
                         "score": round(score, 4), "title": prep["original"][pid]["title"]})
    return rows

def save_csv(rows: list, name: str):
    """Write the rows to results/<name>, floats rounded to 3 decimals."""
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 3) if isinstance(value, float) else value
                             for key, value in row.items()})
    print(f"Saved {RESULTS / name}")


if __name__ == "__main__":
    prep = prepare(preprocess(db_fashion))
    prep["w2v"] = train_word2vec(prep)

    # Item 1: metrics and top 10 of every ranking on the 7 labeled queries
    metrics, top10 = [], []
    for name, ranking in RANKINGS.items():
        metrics += [{"ranking": name, **row} for row in evaluate(prep, ranking)]
        top10 += [{"ranking": name, **row} for row in top_k(prep, ranking, all_queries(), 10)]
    save_csv(metrics, "evaluation.csv")
    save_csv(top10, "rankings_top10.csv")

    print(f"\n{'ranking':20} {'P@10':>6} {'NDCG@10':>8} {'MAP':>6} {'MRR':>6}   (mean of the 7 queries)")
    for row in metrics:
        if row["query_id"] == "MEAN":
            print(f"{row['ranking']:20} {row['P@10']:6.3f} {row['NDCG@10']:8.3f} {row['AP']:6.3f} {row['RR']:6.3f}")

    # Item 2: top 20 of word2vec + cosine for the 5 Part 2 queries (same ids as in the labels: 3 to 7)
    save_csv(top_k(prep, word2vec_cosine, dict(enumerate(TEST_QUERIES, start=3)), 20), "word2vec_top20.csv")
