"""Build Report_Part3.pdf: write the word2vec top-20 appendix from the CSV, then run XeLaTeX twice.

Run after Scoring and Figures: py -m part3.build_report (from the repo root).
"""
import csv
import subprocess
from pathlib import Path

FOLDER = Path(__file__).parent
RESULTS = FOLDER / "results"
LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}


def latex_text(text: str) -> str:
    """Escape LaTeX special characters and turn unusual spaces into normal ones."""
    text = " ".join(text.split())
    return "".join(LATEX_SPECIAL.get(c, c) for c in text)

def write_appendix():
    """results/word2vec_top20.tex: one longtable block per query."""
    with open(RESULTS / "word2vec_top20.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = []
    for row in rows:
        if row["rank"] == "1":
            if lines:
                lines.append(r"\end{longtable}")
            lines += [rf"\subsection*{{Query {row['query_id']}: \texttt{{{latex_text(row['query'])}}}}}",
                      r"\begin{longtable}{r r l p{0.6\textwidth}}", r"\toprule",
                      r"Rank & Cosine & PID & Title \\", r"\midrule", r"\endhead"]
        lines.append(rf"{row['rank']} & {float(row['score']):.3f} & \texttt{{{row['pid']}}} & "
                     rf"{latex_text(row['title'])} \\")
    lines.append(r"\end{longtable}")
    (RESULTS / "word2vec_top20.tex").write_text("\n".join(lines), encoding="utf-8")

def compile_pdf():
    """XeLaTeX twice (the second pass resolves the references)."""
    for _ in range(2):
        subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "Report_Part3.tex"],
                       cwd=FOLDER, check=True, stdout=subprocess.DEVNULL)
    print(f"Built {FOLDER / 'Report_Part3.pdf'}")


if __name__ == "__main__":
    write_appendix()
    compile_pdf()
