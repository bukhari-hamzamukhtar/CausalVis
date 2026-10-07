"""
paper/make_builds.py  —  build the two versions of the paper from one source
===========================================================================

    python paper/make_builds.py            # both
    python paper/make_builds.py cvpr       # the anonymous conference folder only
    python paper/make_builds.py arxiv      # the arXiv source package only

main.tex and supp.tex look for a file named ANONYMOUS beside them, so the conference folder
gets that marker and the arXiv folder does not. Nothing is edited by hand and the two cannot
drift apart. See BUILDS.md for what differs between them.

Neither folder is tracked in git: both are generated, and the style files they need live in
paper/style.
"""

import os
import shutil
import sys
import zipfile

PAPER = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(PAPER, "style")
FIGS = [f for f in sorted(os.listdir(os.path.join(PAPER, "figures"))) if f.endswith(".pdf")]


def fresh(name):
    out = os.path.join(PAPER, name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "figures"))
    for f in FIGS:
        shutil.copy2(os.path.join(PAPER, "figures", f), os.path.join(out, "figures", f))
    return out


def build_cvpr():
    """Anonymous, official template, supplement kept as its own document."""
    out = fresh("cvpr")
    for f in ("main.tex", "supp.tex", "refs.bib"):
        shutil.copy2(os.path.join(PAPER, f), os.path.join(out, f))
    for f in ("cvpr.sty", "ieeenat_fullname.bst"):
        shutil.copy2(os.path.join(STYLE, f), os.path.join(out, f))
    with open(os.path.join(out, "ANONYMOUS"), "w") as fh:
        fh.write("This file makes main.tex and supp.tex build the anonymous version.\n")
    print("built", out)
    print("  compile: pdflatex main; bibtex main; pdflatex main; pdflatex main; "
          "pdflatex supp; pdflatex supp")
    return out


def build_arxiv():
    """Named, supplement appended, ready to upload as source.

    arXiv does not run BibTeX, so the compiled main.bbl ships with it. cvpr.sty is left out
    on purpose: it carries its own copy of eso-pic and clashes with pdfpages, which this
    build needs to append the supplement.
    """
    out = fresh("arxiv")
    for f in ("main.tex", "main.bbl", "supp.pdf"):
        src = os.path.join(PAPER, f)
        if not os.path.exists(src):
            raise SystemExit("missing %s: compile the paper in paper/ first" % f)
        shutil.copy2(src, os.path.join(out, f))
    shutil.copy2(os.path.join(STYLE, "ieeenat_fullname.bst"),
                 os.path.join(out, "ieeenat_fullname.bst"))
    z = os.path.join(PAPER, "arxiv_submission.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, _, files in os.walk(out):
            for f in files:
                full = os.path.join(base, f)
                zf.write(full, os.path.relpath(full, out).replace("\\", "/"))
    print("built", out)
    print("  package: %s  %.1f MB" % (z, os.path.getsize(z) / 1e6))
    return out


def build_tmlr():
    """Anonymous, single column in the TMLR style, appendix inside the same PDF.

    TMLR rejects a non-anonymous submission without review, and the paper must not link to
    any version that carries the authors' names, so the repository link stays hidden here
    too. Supplementary files may be uploaded separately as well, up to 100 MB.
    """
    out = fresh("tmlr")
    for f in ("main.tex", "supp.tex", "refs.bib"):
        shutil.copy2(os.path.join(PAPER, f), os.path.join(out, f))
    for f in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty"):
        shutil.copy2(os.path.join(STYLE, f), os.path.join(out, f))
    for name, why in (("ANONYMOUS", "double blind"), ("TMLR", "single column, TMLR style")):
        with open(os.path.join(out, name), "w") as fh:
            fh.write(why + "\n")
    supp = os.path.join(PAPER, "supp.pdf")
    if os.path.exists(supp):
        shutil.copy2(supp, os.path.join(out, "supp.pdf"))
    print("built", out)
    print("  compile: pdflatex supp; pdflatex supp; "
          "pdflatex main; bibtex main; pdflatex main; pdflatex main")
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["cvpr", "tmlr", "arxiv"]
    if "cvpr" in which:
        build_cvpr()
    if "tmlr" in which:
        build_tmlr()
    if "arxiv" in which:
        build_arxiv()
