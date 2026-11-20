"""Evidence and figures for the Part 3 report. Run after Scoring: py -m part3.Figures (from the repo root).

Reads results/evaluation.csv, rankings_top10.csv and word2vec_top20.csv, recomputes what needs the products
(lengths, ratings, stock), and writes results/*.csv plus results/figures/*.png.
"""
import csv
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part1.DataPreparation import db_fashion, preprocess, preprocess_text
from part3.Base import prepare, search, matching, all_queries, VALIDATION_LABELS, TEST_LABELS
from part3.Scoring import RANKINGS, RESULTS, save_csv, train_word2vec

FIGURES = RESULTS / "figures"
NAMES = list(RANKINGS)                       # fixed order: the colour follows the ranking
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def read(name: str) -> list:
    """Rows of results/<name>."""
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def labels() -> dict:
    """{(query_id, pid): "1" or "0"} for every labeled product."""
    found = {}
    for path in (VALIDATION_LABELS, TEST_LABELS):
        with open(path, newline="", encoding="utf-8-sig") as f:
            found |= {(int(r["query_id"]), r["pid"]): r["labels"].strip() for r in csv.DictReader(f)}
    return found

def top10_by_ranking() -> dict:
    """{ranking: {query_id: [pids, best first]}}."""
    tops = defaultdict(lambda: defaultdict(list))
    for r in read("rankings_top10.csv"):
        tops[r["ranking"]][int(r["query_id"])].append(r["pid"])
    return tops

def save_figure(fig, name: str):
    """Write results/figures/<name>."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURES / name}")


# ---------- evidence (CSV) ----------

def judgements(tops: dict) -> list:
    """Per ranking and query: products of the top 10 labeled relevant, not relevant, or never judged."""
    found = labels()
    rows = []
    for name in NAMES:
        for q, pids in tops[name].items():
            marks = [found.get((q, pid)) for pid in pids]
            rows.append({"ranking": name, "query_id": q, "relevant": marks.count("1"),
                         "not_relevant": marks.count("0"), "unjudged": marks.count(None)})
    return rows

def overlap(tops: dict) -> list:
    """Mean number of shared products between the top 10 of two rankings, over the 7 queries."""
    rows = []
    for a in NAMES:
        for b in NAMES:
            shared = [len(set(tops[a][q]) & set(tops[b][q])) for q in tops[a]]
            rows.append({"ranking_a": a, "ranking_b": b, "mean_shared_top10": mean(shared)})
    return rows

def top10_profile(tops: dict, prep: dict) -> list:
    """Per ranking and query: mean length (tokens), mean rating, unrated and out-of-stock products in the top 10."""
    rows = []
    for name in NAMES:
        for q, pids in tops[name].items():
            ratings = [prep["numeric"][p]["average_rating"] for p in pids]
            rows.append({"ranking": name, "query_id": q,
                         "mean_length": mean(prep["lengths"][p] for p in pids),
                         "mean_rating": mean(r for r in ratings if r is not None) if any(ratings) else None,
                         "unrated": sum(r is None for r in ratings),
                         "out_of_stock": sum(bool(prep["numeric"][p]["out_of_stock"]) for p in pids)})
    return rows

def corpus_numeric(prep: dict) -> list:
    """Corpus figures quoted in the report: size, mean length, mean rating, unrated and out-of-stock products."""
    numeric = prep["numeric"].values()
    return [{"products": len(prep["lengths"]), "mean_length": prep["avg_length"],
             "mean_rating": mean(n["average_rating"] for n in numeric if n["average_rating"] is not None),
             "unrated": sum(n["average_rating"] is None for n in numeric),
             "out_of_stock": sum(bool(n["out_of_stock"]) for n in numeric)}]

def queries(prep: dict) -> list:
    """The 7 queries: preprocessed terms and number of AND matches."""
    rows = []
    for q, text in all_queries().items():
        terms = preprocess_text(text)
        rows.append({"query_id": q, "query": text, "terms": " ".join(terms),
                     "and_matches": len(matching(terms, prep["index"]))})
    return rows

def first_results(prep: dict) -> list:
    """First result of every ranking for every query: its length and the tf of each query term."""
    rows = []
    for name, ranking in RANKINGS.items():
        for q, text in all_queries().items():
            terms = preprocess_text(text)
            pid = search(text, prep, ranking)[0][0]
            rows.append({"ranking": name, "query_id": q, "pid": pid, "length": prep["lengths"][pid],
                         "tf": " ".join(str(prep["index"][t]["postings"][pid]) for t in terms),
                         "title": prep["original"][pid]["title"]})
    return rows

def score_effect(tops: dict, prep: dict) -> list:
    """Products of the cosine top 10 that our score pushes out of it, with their rating and stock."""
    rows = []
    for q, pids in tops["tf_idf_cosine"].items():
        for pid in pids:
            if pid not in tops["tikitikiscore"][q]:
                numeric = prep["numeric"][pid]
                rows.append({"query_id": q, "pid": pid, "rating": numeric["average_rating"],
                             "out_of_stock": bool(numeric["out_of_stock"]),
                             "title": prep["original"][pid]["title"]})
    return rows


# ---------- figures ----------

def plot_metrics(evaluation: list, judged: list):
    """Grouped bars: mean P@10, NDCG@10, MAP, MRR per ranking; right panel: what the 70 top-10 results are."""
    means = {r["ranking"]: r for r in evaluation if r["query_id"] == "MEAN"}
    metrics = [("P@10", "P@10"), ("NDCG@10", "NDCG@10"), ("AP", "MAP"), ("RR", "MRR")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.8),
                                  gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.55})
    width = 0.16
    for i, name in enumerate(NAMES):
        xs = [m + (i - 2) * width for m in range(len(metrics))]
        ax.bar(xs, [float(means[name][key]) for key, _ in metrics], width * 0.88,
               color=COLORS[i], label=name, zorder=2)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("mean over the 7 queries")
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")

    # greys, not series colours: on the left the colours already mean "which ranking"
    parts = [("relevant", "relevant", "#52514e", ""), ("not_relevant", "not relevant", "#a8a69e", "////"),
             ("unjudged", "never judged", "#e4e3df", "")]
    left = [0] * len(NAMES)
    for key, label, color, hatch in parts:
        counts = [sum(r[key] for r in judged if r["ranking"] == name) for name in NAMES]
        ax2.barh(range(len(NAMES)), counts, left=left, color=color, height=0.7, label=label,
                 hatch=hatch, edgecolor="white", linewidth=1, zorder=2)
        for y, (c, l) in enumerate(zip(counts, left)):
            if c >= 6:
                ax2.text(l + c / 2, y, str(c), ha="center", va="center", fontsize=7,
                         color="white" if key != "unjudged" else INK)
        left = [l + c for l, c in zip(left, counts)]
    ax2.set_yticks(range(len(NAMES)), NAMES, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 70)
    ax2.set_xticks([0, 35, 70])
    ax2.set_title("the 70 top-10 results", fontsize=8, color=INK)
    ax2.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3,
               handlelength=1, columnspacing=0.8)
    save_figure(fig, "metrics.png")

def plot_overlap(rows: list):
    """Heatmap (one hue, light to dark) of the mean shared products between top-10 lists."""
    matrix = [[next(r["mean_shared_top10"] for r in rows if r["ranking_a"] == a and r["ranking_b"] == b)
               for b in NAMES] for a in NAMES]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=10)
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            ax.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i][j] > 6 else INK)
    ax.set_xticks(range(len(NAMES)), NAMES, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(NAMES)), NAMES, fontsize=7)
    ax.spines[:].set_visible(False)
    ax.set_title("products shared by two top-10 lists (mean of 7 queries)", fontsize=8, color=INK)
    save_figure(fig, "overlap.png")

def plot_lengths(profile: list, avg_length: float):
    """Dot plot: mean length of the top 10 per query, one marker per ranking; dashed line = corpus mean."""
    queries = all_queries()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, name in enumerate(NAMES):
        rows = sorted((r for r in profile if r["ranking"] == name), key=lambda r: r["query_id"])
        ys = [r["query_id"] + (i - 2) * 0.12 for r in rows]
        ax.scatter([r["mean_length"] for r in rows], ys, s=22, color=COLORS[i], label=name,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(avg_length, color=MUTED, linestyle="--", linewidth=1, zorder=1)
    ax.text(avg_length, 0.35, f" corpus mean {avg_length:.0f}", fontsize=7, color=MUTED, va="center")
    ax.set_yticks(list(queries), [f"{q}. {text}" for q, text in queries.items()], fontsize=7)
    ax.set_ylim(7.5, 0.2)
    ax.set_xlabel("mean length of the top-10 products (tokens)")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save_figure(fig, "lengths.png")

def plot_word2vec(rows: list):
    """Score against rank for each of the 5 test queries (word2vec + cosine)."""
    by_query = defaultdict(list)
    for r in rows:
        by_query[(int(r["query_id"]), r["query"])].append(float(r["score"]))
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for i, ((q, text), scores) in enumerate(sorted(by_query.items())):
        ax.plot(range(1, len(scores) + 1), scores, color=COLORS[i], linewidth=2, label=f"{q}. {text}")
    ax.set_xlabel("rank")
    ax.set_ylabel("cosine (query, product)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    save_figure(fig, "word2vec_scores.png")


if __name__ == "__main__":
    prep = prepare(preprocess(db_fashion))
    prep["w2v"] = train_word2vec(prep)   # first_results runs every ranking, word2vec included
    tops = top10_by_ranking()

    judged = judgements(tops)
    shared = overlap(tops)
    profile = top10_profile(tops, prep)
    save_csv(judged, "top10_judgements.csv")
    save_csv(shared, "top10_overlap.csv")
    save_csv(profile, "top10_profile.csv")
    save_csv(corpus_numeric(prep), "corpus_numeric.csv")
    save_csv(queries(prep), "queries.csv")
    save_csv(first_results(prep), "first_results.csv")
    save_csv(score_effect(tops, prep), "score_effect.csv")

    plot_metrics(read("evaluation.csv"), judged)
    plot_overlap(shared)
    plot_lengths(profile, prep["avg_length"])
    plot_word2vec(read("word2vec_top20.csv"))
