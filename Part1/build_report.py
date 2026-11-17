import pypandoc
pypandoc.convert_file("Report_Part1.md","pdf",outputfile="Report_Part1.pdf",extra_args=[
    "--pdf-engine=xelatex","--toc","--shift-heading-level-by=-1","-V","toc-depth=2","-V","geometry:margin=1.7cm","-V","fontsize=10pt","-V","colorlinks=true","-V","linkcolor=blue","-V","toccolor=blue",
    "-V", r"header-includes=\usepackage{float}\floatplacement{figure}{H}",  # figures stay next to their text
])
print("Report_Part1.pdf written")
