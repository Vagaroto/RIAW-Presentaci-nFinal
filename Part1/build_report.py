"""Build Report_Part1.pdf from Report_Part1.md with pandoc and XeLaTeX.

Run from Part1/ so that the figure paths (eda_results/figures/...) resolve.
"""
import pypandoc

pypandoc.convert_file(
    "Report_Part1.md",
    "pdf",
    outputfile="Report_Part1.pdf",
    extra_args=[
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=2cm",
        "-V", "fontsize=10pt",
        "-V", "colorlinks=true",
        "-V", "mainfont=Cambria",
        "-V", "monofont=Consolas",
        "-V", r"header-includes=\usepackage{float}\floatplacement{figure}{H}",  # figures stay next to their text
    ],
)
print("Report_Part1.pdf written")
