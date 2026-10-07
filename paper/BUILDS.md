# Three builds from one source

`main.tex` and `supp.tex` look for marker files beside them: `ANONYMOUS` hides the author
block, the acknowledgements and the repository link, and `TMLR` switches to one column in the
journal's own style. Nothing else is edited by hand, so the versions cannot drift apart.

| | `paper/` (arXiv) | `paper/tmlr/` (journal submission) | `paper/cvpr/` (conference) |
|---|---|---|---|
| marker files | none | `ANONYMOUS`, `TMLR` | `ANONYMOUS` |
| author block | name and institute | hidden by `tmlr.sty` | "Anonymous CVPR submission" |
| layout | two columns, CVPR-like geometry | one column, 6.5in, `tmlr.sty` | two columns, official `cvpr.sty`, review mode |
| citations | numbered | author and year | numbered |
| repository link in the abstract | the GitHub URL | "will be released" | "will be released" |
| acknowledgements and the assistance statement | printed | omitted | omitted |
| supplement | appended | appended as an appendix | separate `supp.pdf`, uploaded on its own |
| pages | 19 | 22 | 8 of content, references to 10 |

## Rebuilding

```bash
python paper/make_builds.py        # regenerates all three folders
```

Then in each folder: `pdflatex supp; pdflatex supp` first where a `supp.tex` is present, then
`pdflatex main; bibtex main; pdflatex main; pdflatex main`. The supplement has to be compiled
before the paper in the arXiv and TMLR builds, because they include its PDF.

`paper/arxiv_submission.zip` is built by the packaging script and holds `main.tex`,
`main.bbl`, `supp.pdf`, the bibliography style and the nine figures. arXiv does not run BibTeX,
which is why the compiled `.bbl` ships with it.

## Why `cvpr.sty` lives only in `paper/cvpr/`

It carries its own inlined copy of `eso-pic`, which clashes with `pdfpages`. The arXiv build
needs `pdfpages` to append the supplement, so it uses the fallback geometry instead. The CVPR
build does not append anything, so it loads the official style and no `pdfpages`.

## Before submitting to TMLR

- TMLR rejects a non-anonymous submission without review, and the paper must not link to any
  version that carries the authors' names. The build already removes the repository link and
  the acknowledgements; keep it that way and post the arXiv version separately.
- Any length is allowed, and the appendix belongs inside the same PDF after the references,
  which is what this build produces. Supplementary files may also be uploaded, up to 100 MB,
  PDF or ZIP, and they must be anonymous too.
- Submission is through OpenReview, and authors recommend an action editor.
- Everything is licensed CC BY 4.0 from submission onward, with copyright kept by the author.

## Before submitting to CVPR

- Put the real paper ID in `\def\paperID{*****}` once OpenReview assigns one.
- Re-read the CVPR 2027 LLM section: at the time of writing it says it is still being
  finalised and will be updated before the deadline.
- Check that no figure, table or sentence names the repository, the institute or the author.
  A quick check: `pdftotext main.pdf - | grep -i "bukhari\|giki\|ghulam\|github"` must return
  nothing.
