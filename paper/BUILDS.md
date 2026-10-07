# Two builds from one source

`main.tex` and `supp.tex` both look for a file named `ANONYMOUS` beside them. Its presence
selects the double-blind conference build. Nothing else has to be edited, so the two versions
cannot drift apart.

| | `paper/` (arXiv, journal) | `paper/cvpr/` (conference submission) |
|---|---|---|
| `ANONYMOUS` file | absent | present |
| author block | name and institute | "Anonymous CVPR submission" |
| style | fallback geometry that matches CVPR | official `cvpr.sty`, review mode |
| repository link in the abstract | the GitHub URL | "will be released" |
| acknowledgements and the assistance statement | printed | omitted, they break anonymity |
| supplement | appended, one PDF of 19 pages | separate `supp.pdf`, uploaded on its own |
| page count | 19 | 8 pages of content, references to page 10 |

## Rebuilding

```bash
cd paper        && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd paper/cvpr   && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd paper/cvpr   && pdflatex supp && pdflatex supp
```

`paper/arxiv_submission.zip` is built by the packaging script and holds `main.tex`,
`main.bbl`, `supp.pdf`, the bibliography style and the nine figures. arXiv does not run BibTeX,
which is why the compiled `.bbl` ships with it.

## Why `cvpr.sty` lives only in `paper/cvpr/`

It carries its own inlined copy of `eso-pic`, which clashes with `pdfpages`. The arXiv build
needs `pdfpages` to append the supplement, so it uses the fallback geometry instead. The CVPR
build does not append anything, so it loads the official style and no `pdfpages`.

## Before submitting to CVPR

- Put the real paper ID in `\def\paperID{*****}` once OpenReview assigns one.
- Re-read the CVPR 2027 LLM section: at the time of writing it says it is still being
  finalised and will be updated before the deadline.
- Check that no figure, table or sentence names the repository, the institute or the author.
  A quick check: `pdftotext main.pdf - | grep -i "bukhari\|giki\|ghulam\|github"` must return
  nothing.
