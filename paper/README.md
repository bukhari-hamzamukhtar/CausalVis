# The paper

| file | what it is |
|---|---|
| `main.tex` | the paper (CVPR-sized layout, 8 pages without references) |
| `supp.tex` | the supplementary material (sections A to J) |
| `refs.bib` | 57 references, each checked against arXiv, Crossref or OpenAlex on 20 Sep 2026 |
| `figures/` | the nine figures, vector PDF, Arial embedded, no Type 3 fonts |
| `main.pdf`, `supp.pdf` | the compiled paper (8 pages plus references) and supplement (8 pages) |
| `causalvis_paper_overleaf.zip` | everything above, ready to upload to Overleaf |

## How to compile

On this laptop (MiKTeX is installed under `AppData/Local/Programs/MiKTeX`):

    cd paper
    pdflatex main && bibtex main && pdflatex main && pdflatex main
    pdflatex supp && pdflatex supp

Redraw the figures from the data with:

    python paper_exp/figures/make_figures.py

On Overleaf: new project, upload `causalvis_paper_overleaf.zip`, compile `main.tex`. For the
real CVPR layout, open the CVPR template first and keep its `cvpr.sty` and
`ieeenat_fullname.bst`; `main.tex` picks them up on its own and falls back to a CVPR-sized
layout when they are missing, which is how the current PDF was built.

## State
Text, numbers, figures and references are final. The body fits 8 pages and the references
follow, which is what CVPR asks for.

## Where it goes first
TMLR and arXiv, because they cost nothing (see paper_exp/VENUES.md). For TMLR the content
stays the same; the template becomes single column and citations become author-year
(`\citet` and `\citep`). That switch takes about fifteen minutes.

## Before submitting to a double-blind venue (CVPR, NeurIPS)
- Set `\anontrue` in main.tex, and remove names, the GitHub link and any "our previous work".
  Code goes in the supplement as an anonymous zip, or as an anonymous link.
- Check the page count again with the official `cvpr.sty`, which differs from our fallback by
  a few lines per page.
- The supplement is a separate PDF, uploaded by the supplementary deadline.

## If it goes to NeurIPS or TMLR later
- TMLR and NeurIPS use author-year citations. Replace narrative citations such as
  `Ishay~\etal~\cite{x}` with `\citet{x}` and the others with `\citep{x}`.
- Re-export the full-width figures at the new text width (see FIGURES.md, section 0.1).

## Other title options
1. Better Physics, Same Score: What an Auditable Counterfactual Engine Reveals About CLEVRER (current)
2. What Does a Counterfactual Video Benchmark Measure? An Audit with an Evidence-Returning Engine
3. Right for the Right Reason? Auditing Counterfactual Video Reasoning on CLEVRER
