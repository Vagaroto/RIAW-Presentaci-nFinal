# Part 3 — Ranking & Filtering: step-by-step build guide

What the assignment asks for (`IRWA_project_part_3.pdf`):

1. An **AND search** followed by 3 rankers: **TF-IDF + cosine**, **BM25**, and **your own score** (built on the numeric fields), with an explanation of how TF-IDF and BM25 differ and the pros and cons of your score.
2. A **word2vec + cosine** ranker, with the **top 20 for each of the 5 test queries** of Part 2.
3. A written answer to: *is there a better representation than word2vec?* (doc2vec, sentence embeddings…)

---

## Step 0 — Setup and what you reuse

**0.1 Create your package.** In the repo root (`RIAW-Presentaci-nFinal/`), create:

```
MyPart3/
  __init__.py          (empty, makes it importable as MyPart3.xxx)
  pipeline.py          search + evaluation driver
  tfidf_cosine.py      ranker a
  bm25.py              ranker b
  our_score.py         ranker c
  word2vec_cosine.py   ranker for item 2
  results/             CSV outputs (created by the code)
```

**0.2 Python version.** Use `py -3.11` for everything in this part. gensim (needed for word2vec) has no wheel for 3.14, and 3.11 already has pandas, nltk and gensim 4.4. Always run **from the repo root** as a module:

```powershell
cd RIAW-Presentaci-nFinal
py -3.11 -m MyPart3.pipeline
```

(`-m` makes `from Part1... import` and `from Part2... import` work.)

**0.3 What you import, so you don't rewrite it:**

| From | Name | What it gives you |
|---|---|---|
| `Part1.DataPreparation` | `db_fashion` | the raw dataset (loaded on import) |
| | `preprocess(db_fashion)` | list of `doc` dicts (see below) |
| | `preprocess_text(str)` | the **same** normalize → stop words → stem pipeline as the documents. Always use it on the query |
| `Part2.InvertedIndex` | `build_index(docs)` | the inverted index (see below) |
| | `get_query_index(index, query: str)` | list of pids that contain **all** query terms (AND) |
| | `TEST_QUERIES` | your 5 test queries (they become query ids 3–7) |
| `Part2.Evaluation` | `VALIDATION_QUERIES` | `{1: "...", 2: "..."}` |
| | `VALIDATION_LABELS`, `read_labels(path)` | `{query_id: set of relevant pids}` for queries 1–2 |
| | `evaluate_query(ranked_pids, relevant_set)` | dict of metrics: `P@k, R@k, AP@k, F1@k, NDCG@k` for k = 5, 10, 20, plus `AP`, `RR` |
| `Part2.LabelTestQueries` | `CRITERIA` | `{query_id: function(original_record) -> bool}` for queries 3–7 |

**0.4 The two data structures. Learn these by heart:**

```python
doc = {
  "pid": "TKPFCZ9EA7H5FYZH",
  "original":  {...raw record: title, brand, description, product_details, ...},   # show results from here
  "processed": {"title": [...], ..., "text": ["full", "sleev", "solid", "women", ...]},  # indexed tokens
  "numeric":   {"selling_price": 599.0, "actual_price": 999.0, "discount": 40.0,
                "average_rating": 3.9, "out_of_stock": False},   # None when missing
}

index[term] = {"idf": log(N / df_t),            # natural log, already computed
               "postings": [(pid, tf_td), ...]}  # tf = raw count of term in doc["processed"]["text"]
```

> [!NOTE]
> `index` is a plain `dict`. Use `index.get(term)` for a term that may be missing, because `index[term]` raises a `KeyError`.

**0.5 The one design rule: the Ranker interface.** Every ranker file defines the same class:

```python
class Ranker:
    def __init__(self, docs: list, index: dict):
        """Precompute everything that does not depend on the query (runs ONCE)."""

    def score(self, query_terms: list, pids: list) -> dict:
        """query_terms: preprocessed query tokens; pids: the AND matches.
        Returns {pid: score} for EVERY pid in pids (higher = better)."""
```

Why: the pipeline can then swap rankers by name. The AND filter is applied **once, before** ranking, so the rankers only order the matches and never decide which documents match.

---

## Step 1 — `pipeline.py`: the AND search with a dummy ranker

Build the plumbing first, with a ranker that cannot be wrong.

**1.1** In `pipeline.py` write:

| Function | Inputs | Returns | Does |
|---|---|---|---|
| `search(query, ranker, index)` | raw query `str`, a Ranker, the index | `[(pid, score), ...]` sorted **best first** | `pids = get_query_index(index, query)` → `ranker.score(preprocess_text(query), pids)` → sort the items by score, descending |
| `all_queries()` | — | `{1: q1, 2: q2, 3: ..., 7: ...}` | merge `VALIDATION_QUERIES` with `enumerate(TEST_QUERIES, start=3)` |

**1.2** Temporarily define a dummy ranker inside `pipeline.py`:

```python
class ConstantRanker:
    def __init__(self, docs, index): pass
    def score(self, query_terms, pids): return {pid: 1.0 for pid in pids}
```

**1.3** In an `if __name__ == "__main__":` block: `docs = preprocess(db_fashion)`, `index = build_index(docs)`, then print `len(search(q, ConstantRanker(docs, index), index))` for each query.

**Run:** `py -3.11 -m MyPart3.pipeline`. Preprocessing 28,080 products takes a while; that's normal.

✅ **Checkpoint:** match counts per query = **1: 500 · 2: 222 · 3: 1451 · 4: 2046 · 5: 72 · 6: 23 · 7: 24**.
These counts are the same for every ranker, because ranking never changes the set of matches.

---

## Step 2 — `tfidf_cosine.py`: TF-IDF + cosine (IRWA02)

**The math (course notation):**

$$w_{t,d}=(1+\ln tf_{t,d})\cdot \ln\frac{N}{df_t}\quad(tf>0),\qquad w_{t,q}=(1+\ln tf_{t,q})\cdot \ln\frac{N}{df_t}$$

$$\cos(\vec q,\vec d)=\frac{\sum_{t\in q} w_{t,q}\,w_{t,d}}{\lVert\vec q\rVert\;\lVert\vec d\rVert},\qquad \lVert\vec d\rVert=\sqrt{\sum_{t\in d} w_{t,d}^2}$$

> [!IMPORTANT]
> $\lVert\vec d\rVert$ runs over **all the terms of the document**, not only the query terms. This is the most common bug. Because you need every term, you can only compute it by sweeping the **whole index** once, in `__init__`.

**Build:**

| Piece | Signature | Does |
|---|---|---|
| helper | `weight(tf: int, idf: float) -> float` | $(1+\ln tf)\cdot idf$ (use `math.log`, the natural log) |
| `__init__` | `(docs, index)` | store `index`. Loop over every `entry in index.values()` and every `(pid, tf)` in its postings, accumulating `weight(tf, idf)**2` per pid (`defaultdict(float)`). Then `self.doc_norm[pid] = sqrt(sum)` |
| `score` | `(query_terms, pids) -> dict` | 1) `Counter(query_terms)` gives $tf_{t,q}$; build `query_weights` (skip terms not in the index). 2) $\lVert\vec q\rVert$. 3) **term-at-a-time** (the COSINESCORE pseudocode of IRWA02 §6): for each query term, walk its postings and, for pids in `set(pids)`, add `w_q * w_d` to `dot[pid]`. 4) return `dot[pid] / (q_norm * doc_norm[pid])` |

> [!TIP]
> Turn `pids` into a `set` before the loop. `pid in list` is O(n) and makes this very slow.

**Test it** by plugging it into `search` instead of `ConstantRanker` and printing the top 5 titles of query 1.

✅ **Checkpoint:** top score of q1 ≈ **0.549**, product `SWSFYTYMNTBNARUN` ("Full Sleeve Solid Women Sweatshirt"). Every score lies in [0, 1].

*Self-check question:* why does dropping $\lVert\vec q\rVert$ not change the ranking? (IRWA02 exercise 3)

---

## Step 3 — `bm25.py`: BM25 (IRWA06)

**The math:**

$$RSV_d=\sum_{t\in q}\ln\frac{N}{df_t}\cdot\frac{(k_1+1)\,tf_{t,d}}{k_1\big((1-b)+b\,\frac{L_d}{L_{ave}}\big)+tf_{t,d}}$$

- $L_d$ = `len(doc["processed"]["text"])`; $L_{ave}$ = the mean of $L_d$ over all the documents.
- Constants: $k_1 = 1.2$, $b = 0.75$. These are the course defaults ("$k_1\in[1.2,2]$, $b=0.75$"). They are not tuned, because 40 labels are too few (see the project rule in `CLAUDE.md`: *no learned weights*).
  ⚠️ `Part2/InvertedIndex.py` used `K1=1.4, B=0.25`. Pick one pair, use it everywhere, and **say which one in the report**.

**Build:**

| Piece | Signature | Does |
|---|---|---|
| module constants | `K1, B = 1.2, 0.75` | |
| `__init__` | `(docs, index)` | `self.doc_len = {pid: L_d}`, `self.avg_len` |
| `score` | `(query_terms, pids) -> dict` | for each query term with an index entry, walk its postings. For pids in the candidate set, add `idf * (K1+1)*tf / (K1*((1-B) + B*L_d/L_ave) + tf)`. Return a value for every pid |

✅ **Checkpoint:** top score of q1 ≈ **9.88**, q6 ≈ **13.7**. The scores are **not** bounded to [0, 1] (unlike cosine).

*Things to observe for the report* (write them down as you go):
- As $tf\to\infty$ the tf factor saturates at $k_1+1$. Cosine has no saturation (only the log).
- How is length handled? Cosine divides by $\lVert\vec d\rVert$; BM25 uses the $b\cdot L_d/L_{ave}$ term.
- Here every document already contains every query term (AND), so the **idf part is the same for every match**. The two rankers differ only in tf handling and length normalisation. This is the key insight for "explain how the ranking differs".

---

## Step 4 — `our_score.py`: your own score (the creative part)

**Template** (a text-relevance score combined with a query-independent quality factor; IRWA02 §7, "net score" $= \text{relevance} + g(d)$):

$$\text{score}(q,d)=\underbrace{\text{BM25}(q,d)}_{\text{text relevance}}\times\underbrace{g(d)}_{\text{product quality}}\qquad\text{or}\qquad \text{BM25}(q,d)+\lambda\,g(d)$$

**What you know about the numeric fields** (`Part1/data_checks_results/numeric_fields.csv`):

| field | missing | range / notes |
|---|---|---|
| `selling_price` | 2 | 99 – 7999, very skewed, so use **log** or **rank** scaling, never raw |
| `actual_price` | 777 | 150 – 12999 |
| `discount` | 855 | 1 – 87 % |
| `average_rating` | 2261 (8 %) | 1 – 5 |
| `out_of_stock` | 0 | 1644 are `True` |

**Design decisions you must make, and justify in the report:**

1. **Which factors.** Candidates: rating (quality), out of stock (a product the user cannot buy), discount (attractiveness), price (cheap ≠ relevant, so is it a good factor?), and maybe the number of *exact title* matches of the query terms.
2. **Normalise each factor to a comparable scale**, e.g. $\text{rating}/5\in[0,1]$, $\text{discount}/100$, $\ln(\text{price})$ min-max scaled.
3. **How to treat `None`.** Don't drop the product, and don't treat it as 0 stars, which would punish new products. A neutral value (0.5 or the mean) is the usual choice.
4. **Combine multiplicatively or additively?** With a multiplicative factor the text relevance stays dominant: a perfect product cannot jump above a much better text match if $g\in[0.5, 1.3]$. With an additive factor you must calibrate $\lambda$ against the BM25 scale (~5–15 here).
5. **Constants** (e.g. how much a 5★ rating adds, the out-of-stock penalty). Choose them by reasoning, **not by fitting the labels** (project rule). Write down why.

**Build:**

| Piece | Signature | Does |
|---|---|---|
| constants | e.g. `OUT_OF_STOCK_FACTOR`, `RATING_WEIGHT`, … | each one documented in a comment |
| `__init__` | `(docs, index)` | `from MyPart3.bm25 import Ranker as BM25`; `self.bm25 = BM25(docs, index)`; `self.numeric = {pid: doc["numeric"]}` (+ any min/max needed for scaling) |
| `quality` | `(pid) -> float` | your $g(d)$ |
| `score` | `(query_terms, pids) -> dict` | `text = self.bm25.score(...)` and then combine with `quality` |

✅ **Checkpoint:** no fixed number here, because it's your formula. Sanity checks: (a) for a fixed text score, an out-of-stock product ranks below an in-stock one; (b) the top 10 still *look* relevant (print titles + rating + stock). The reference placeholder scored MEAN P@10 0.686 vs BM25 0.700. Don't be surprised if quality factors **lower** the metrics. The labels judge topical relevance only, not "a good product to buy". That is a pro/con worth writing about.

---

## Step 5 — `word2vec_cosine.py`: word2vec + cosine (item 2)

**The math.** For a text with known tokens $w_1,\dots,w_n$ (tokens in the word2vec vocabulary):

$$\vec v_{\text{text}}=\frac{1}{n}\sum_{i=1}^{n}\vec v_{w_i}\in\mathbb R^{100},\qquad \text{score}(q,d)=\cos(\vec v_q,\vec v_d)=\frac{\vec v_q\cdot\vec v_d}{\lVert\vec v_q\rVert\,\lVert\vec v_d\rVert}$$

**Build:**

| Piece | Signature | Does |
|---|---|---|
| constants | `MODEL_PATH` (next to the file, e.g. `os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec.model")`), `PARAMS = dict(vector_size=100, window=5, min_count=2, sg=1, epochs=5, seed=42, workers=1)` | `workers=1` + `seed`: reproducible |
| `train_or_load(docs) -> Word2Vec` | | if the file exists, `Word2Vec.load(MODEL_PATH)`. Otherwise `Word2Vec(sentences=[doc["processed"]["text"] for doc in docs], **PARAMS)`, then `.save(MODEL_PATH)` |
| `average_vector(tokens, vectors) -> np.ndarray` | `vectors` = `model.wv` | mean of `vectors[t]` for `t in vectors`. **If no token is known, return a zero vector** of size `vectors.vector_size` |
| `cosine(a, b) -> float` | | `a @ b / (‖a‖‖b‖)`, and **0.0 if a norm is 0** (avoids a division by zero) |
| `__init__` | `(docs, index)` | `self.vectors = train_or_load(docs).wv`; precompute `self.doc_vector[pid]` for **every** product |
| `score` | `(query_terms, pids) -> dict` | one query vector, then cosine with each candidate's vector |

> [!NOTE]
> Train on the **same preprocessed tokens** as the index (stemmed). The query also goes through `preprocess_text`, so query tokens and model vocabulary match. The first run trains the model (about a minute); later runs load it. Delete the `.model` file to retrain.

Concepts to be able to explain (ask me if they're not clear): what `sg=1` (skip-gram) vs CBOW means, `window`, `min_count`, why the vectors capture "words used in similar contexts".

✅ **Checkpoint:** q1 top score ≈ **0.92** and all the scores are high (0.7–0.95). That is expected, because averaged vectors of texts that share every query word all point in a similar direction. This compression of scores is itself a point for item 3.

---

## Step 6 — `pipeline.py` again: run everything and write the results

**6.1** Replace the dummy with a registry and a loader:

```python
RANKERS = {"tfidf": "MyPart3.tfidf_cosine", "bm25": "MyPart3.bm25",
           "ours": "MyPart3.our_score", "word2vec": "MyPart3.word2vec_cosine"}
```

`load_ranker(name, docs, index)` → `importlib.import_module(RANKERS[name]).Ranker(docs, index)`.

**6.2** `relevant_sets(docs) -> {query_id: set(pids)}`:
- queries 1–2: `read_labels(VALIDATION_LABELS)`. Only 40 products are labelled, and every other product counts as not relevant, so their numbers are pessimistic.
- queries 3–7: for each `query_id, is_relevant in CRITERIA.items()`, keep `{doc["pid"] for doc in docs if is_relevant(doc["original"])}`.

**6.3** `run_all(names, docs, index)`: for each ranker name, build it **once**. For each of the 7 queries:
- `ranked = search(query, ranker, index)`
- append the top 20 rows to `top_rows`: `ranker, query_id, query, rank, pid, score, title` (title from `doc["original"]["title"]`)
- `metrics = evaluate_query([pid for pid, _ in ranked], relevant[query_id])` → append to `metric_rows` with `ranker, query_id, query, matches`
- print one line per query plus the MEAN of `P@10, NDCG@10, AP, RR`

**6.4** `write_csv(path, rows)` with `csv.DictWriter` (round floats to 3 decimals). Write `MyPart3/results/top20.csv` and `MyPart3/results/evaluation.csv` (`os.makedirs(..., exist_ok=True)`).

**6.5 (optional, handy)** `argparse` with `--rankers` (subset) and `--query "free text"` (print the top 10 of one query for each ranker).

**Run:**

```powershell
py -3.11 -m MyPart3.pipeline                          # everything
py -3.11 -m MyPart3.pipeline --rankers bm25 tfidf     # if you added argparse
```

✅ **Checkpoint (MEAN over the 7 queries, reference version):**

| ranker | P@10 | NDCG@10 | AP | RR |
|---|---|---|---|---|
| tfidf | 0.700 | 0.689 | 0.566 | 0.736 |
| bm25 | 0.700 | 0.699 | 0.570 | 0.788 |
| word2vec | 0.686 | 0.673 | 0.560 | 0.691 |

Small differences are fine if you chose different constants. A big one (e.g. TF-IDF scores > 1, or BM25 ≈ TF-IDF everywhere) means a bug.

---

## Step 7 — Analysis for the report (no code required, a little optional)

**7.1 TF-IDF vs BM25 (item 1).** Evidence to collect from `top20.csv`:
- **Overlap**: how many of the top 20 pids are shared per query (Jaccard)? Do the **orders** differ (e.g. a few pids with very different ranks)? Write a small helper if you want: `overlap(top20_df, query_id, "tfidf", "bm25")`.
- Pick 1–2 products that move a lot and explain **why** using the formulas (long description? a query term repeated many times?).
- Pros/cons to discuss: saturation, length normalisation, extra parameters ($k_1$, $b$) vs parameter-free cosine, and BM25 as the standard baseline (IRWA06).
- Use the metrics, but remember: 40 labels for q1–2, and rule-based labels for q3–7, so small differences are noise.

**7.2 Your score.** The formula, each factor and constant with its reason, the treatment of missing values, and what it fixes vs. what it breaks (e.g. it pushes popular products over more exact matches, and the labels can't measure "quality").

**7.3 word2vec.** The top-20 tables for queries 3–7 (from `top20.csv`, `ranker == "word2vec"`, `query_id >= 3`), and observations (score compression, synonyms/related words, products where averaging hurts).

**7.4 Better representations (item 3).** Questions your answer should cover:
- Averaging loses **word order** and gives equal weight to every word. What does a TF-IDF-*weighted* average fix?
- **Doc2Vec** (PV-DM / PV-DBOW): learns a vector per document. Pros/cons? (Hint: what happens with a *new* query or document?)
- **Sentence embeddings** (Sentence-BERT and other transformer encoders): contextual meaning and pretraining on huge corpora, vs. cost, domain mismatch (fashion jargon, brand names) and the need for a GPU or an ANN index at scale.
- Which one would you pick for *this* catalogue (short, keyword-like titles), and why?

Use the `researcher` agent or ask me to check facts for 7.4 before you write them.

---

## Final checklist

- [ ] `MyPart3/` runs end-to-end with `py -3.11 -m MyPart3.pipeline`
- [ ] match counts = 500 / 222 / 1451 / 2046 / 72 / 23 / 24
- [ ] TF-IDF scores in [0, 1], with the doc norm over **all** terms
- [ ] BM25 constants chosen and stated
- [ ] your score: formula + justification of each constant + handling of `None`
- [ ] word2vec top 20 for queries 3–7 in `results/top20.csv`
- [ ] report: TF-IDF vs BM25 comparison, pros/cons of your score, the answer to item 3
- [ ] compare with `Part3/` (the reference) only at the end, and decide which version you hand in
