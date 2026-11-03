from part1.DataPreparation import preprocess_text, preprocess
import math
import csv
from collections import defaultdict  

K1= 1.4
B=0.25

# Queries 1 and 2 are the validation queries (validation_labels.csv); these are our own, chosen from term_stats.csv
TEST_QUERIES = ["men polo t-shirt","men round neck half sleeve cotton t-shirt","reebok men t-shirt","men raincoat","white linen shirt"]


def build_index(docs: list):
    # defaultdict(list): accessing a missing key automatically creates it with an empty list []
    inverted_index = defaultdict(list)
    # inverted_index[term] = [(pid, number of times 'term' appears in document 'pid')] tuples in a list

    for doc in docs:
        pid = doc['pid']
        for term in doc['processed']['text']:
            # Add the pid only once per term: docs are processed in order,
            # so if the last pid in the list is this doc, it was already added
            postings = inverted_index[term]
            if postings and postings[-1][0] == pid:
                postings[-1] = (pid, postings[-1][1] + 1)
            else:
                postings.append((pid, 1))

    # Each term gets its idf next to its postings:
    # index[term] = {"idf": float, "postings": [(pid, tf), ...]}
    # A regular dict, so looking up a missing term later doesn't silently add it
    N = len(docs)
    return {term: {"idf": math.log(N / len(postings)),  # len(postings) = df
                   "postings": postings}
            for term, postings in inverted_index.items()}

def get_query_index(inverted_index: dict, query: str) -> list:
    clean_query = preprocess_text(query) 
    result = None

    for q_term in clean_query:
        # 1) Get the pids of the docs containing this term (empty if the term is not in the index)
        entry = inverted_index.get(q_term)
        term_docs = {pid for pid, _ in entry["postings"]} if entry else set()

        # 2) AND: keep only the docs that also contain all previous terms (set intersection)
        result = term_docs if result is None else result & term_docs

    return list(result or [])

def export_term_stats(inverted_index: dict, path="term_stats.csv"):
    """Write one row per term (df, total tf, idf, tf*idf), most popular first, to pick queries by hand."""
    rows = []
    for term, entry in inverted_index.items():
        tf = sum(count for _, count in entry["postings"])
        rows.append((term, len(entry["postings"]), tf, entry["idf"], tf * entry["idf"]))
    rows.sort(key=lambda row: row[4], reverse=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "df", "tf", "idf", "tf_idf"])
        for term, df, tf, idf, tf_idf in rows:
            writer.writerow([term, df, tf, round(idf, 3), round(tf_idf, 3)])

# falta hacer el ranking de los queries, crear las queries según lo que piden con una funcion
#esta hecha la funcion que crea el vocab, que crea el index, que mira el tf de cada term y el que mira el idf
# checkear si eso funciona y hacer lo siguiente

def get_l_ave(N: int, docs: list):
    l_ave = 0
    for doc in docs:
        l_ave+=len(doc['processed']['text'])
    l_ave = l_ave / N
    return l_ave

def tf_idf(query: list, inverted_index: dict):
    """Rank docs by the sum over query terms of tf * idf (a term missing from the index adds nothing)."""
    scores = defaultdict(float)
    for term in query:
        entry = inverted_index.get(term)
        if entry:
            for pid, tf in entry['postings']:
                scores[pid] += tf * entry['idf']
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def search(inverted_index: dict, query: str) -> list:
    """The ranking of the search engine: AND matches ordered by TF-IDF, as (pid, score)."""
    matches = set(get_query_index(inverted_index, query))
    return [(pid, score) for pid, score in tf_idf(preprocess_text(query), inverted_index) if pid in matches]

def bm25(query: list, inverted_index: dict, docs: list, N: int):
    """EXTRA, not the main ranking: BM25 with length normalization, kept to compare with TF-IDF."""
    l_ave = get_l_ave(N, docs)
    scores = defaultdict(float)
    doc_len = {doc['pid']: len(doc['processed']['text']) for doc in docs}   # pid -> length

    for term in query:
        for pid, tf in inverted_index[term]['postings']:
            l_d = doc_len[pid]
            scores[pid] += inverted_index[term]['idf'] * ((K1+1)*tf) / (K1*((1-B)+B*(l_d/l_ave)) + tf)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    from part1.DataPreparation import db_fashion

    docs = preprocess(db_fashion)
    inverted_index = build_index(docs)
    titles = {doc['pid']: doc['original']['title'] for doc in docs}

    for number, query in enumerate(TEST_QUERIES, start=3):   # 1 and 2 are the validation queries
        ranked_results = search(inverted_index, query)   # TF-IDF over the AND matches
        print(f"\nQuery {number}: {query} -> {len(ranked_results)} results")
        for pid, score in ranked_results[:5]:
            print(f"  {score:.3f}  {pid}  {titles[pid]}")

        if ranked_results:   # extra: BM25 over the same matches, only to compare
            matches = {pid for pid, _ in ranked_results}
            bm25_scores = bm25(preprocess_text(query), inverted_index, docs, len(docs))
            print("  [extra] BM25 top 3:")
            for pid, score in [r for r in bm25_scores if r[0] in matches][:3]:
                print(f"    {score:.3f}  {pid}  {titles[pid]}")
