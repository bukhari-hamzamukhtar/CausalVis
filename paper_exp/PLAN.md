# Paper preparation: everything to finish before writing

Started 2026-09-18.

## Evaluation sets
| name   | videos | counterfactual q | role |
|--------|--------|------------------|------|
| VAL-A  | 500 CLEVRER-validation videos in our val split | 941 | development, all selection |
| TEST-A | 500 CLEVRER-validation videos in our test split | 930 | earlier test (consulted twice for v7.3) |
| TEST-B | 898 CLEVRER-train videos in our test split | 1,689 | FRESH: never trained on, never scored; pre-registered |
| VAL-B  | 898 CLEVRER-train videos in our val split | 1,672 | spare dev data |
World model, laws and entry profile never saw val/test trajectories; no question/answer
was ever used for training. The published CLEVRER test server closed 2026-01-16.

## Single code path
paper_exp/make_final.py -> paper_exp/cf_world.py + evaluate.py (knobs SIM, CF_LOOKBACK,
NO_VOXEL, NO_FORCE, DRAG_SCALE, --model, --no-entry-fix, --no-sim).
Identity checks passed: learned lb3 = v7_3 val_lb3 (104/104); laws lb12 + entry = v8
val_entry (104/104). Old no-sim records predate the kick detectors -> no-sim rerun.

## Work items
| # | item | status |
|---|------|--------|
| A0 | detection-only tracks (no annotation at test time): paper_exp/detonly.py; inventory recall 99.6% / precision 99.0% on 200 TRAIN clips | done |
| A1 | fair learned vs laws vs straight vs straight+friction vs no physics, same lookback sweep {1,3,6,12} on VAL-A (detection-only tracks, parsed programs) | done (TEST: paper_exp/final_report.txt) |
| A2 | component ablations at the chosen setting (TEST-A/B only, pre-registered) | done |
| A3 | simulator-quality audit: physics_quality.py (rollout error, shadow replay) for every simulator; benchmark on TEST-A/B | done |
| A4 | PREREGISTRATION.md written; TEST-A + TEST-B once for all listed systems | done |
| A5 | perception parity (perception_parity.py) | done |
| A6 | bootstrap CIs, McNemar, subset tables, error budget for final systems | done |
| A7 | seeds 1, 2 (stopped after 2 of 4 epochs by a power loss; best = epoch 0 as in the original) | done |
| A8 | question parser qparser.py (TRAIN templates): val exact-match CF 100%, predictive 100%, explanatory 99.995%, descriptive 99.80% | done |
| B1 | executor.py (official NS-DR semantics) + eval_qtypes.py + score_qtypes.py; smoke test OK | done (TEST A+B) |
| C1 | downloads approved 2026-09-18. CLEVRER-Humans: humans.py (LLM grounding + trace / but-for), grounding running. ComPhy: annotations downloading. CoPhy: 18 GB downloading to C:/cophy_dl (slow) | done (CLEVRER-Humans, ComPhy val, CoPhy BallsCF) |
| D1 | REPRODUCE.md written; pushing the code was a later decision | partly |
| D2 | CLEVRER authors: emailed twice, four months apart, no reply. DROPPED; the paper states that the server is closed and that requests went unanswered | closed |
| E1 | evidence audit of CLEVRER answers (audit_traces.py): 98.2% of checkable correct answers rest on a real event | done |
| E2 | explanatory questions by the but-for test (butfor_explanatory.py): 77.7% vs chain rule 88.4% on TEST A+B | done |
| E3 | model-free key consistency (key_consistency.py): "responsible" contradicts CLEVRER's own counterfactual key in 35.3% of cases; pair collides once in the video in all 113, same collision in >= 56 | done |
| E4 | CoPhy event audit (2/4/6 balls: P 0.85/0.61/0.49, R 0.65/0.49/0.41); ComPhy predictive evidence 97.7% valid | done |
| F1 | FIGURES.md: every figure specified; data in figures/data, renders in figures/src | done |
| F2 | SETTINGS.md, VENUES.md, LIMITATIONS_IMPACT.md, README.md, .gitignore release rules | done (nothing committed) |
| F3 | clean single-process runtime: learned 2.24 s, laws 1.36 s per question | done |
| W1 | paper draft: paper/main.tex, supp.tex, refs.bib (57 verified refs) | done; figures pending |
| F4 | figures drawn (hand drafts -> paper_exp/figures/make_figures.py, data-driven); 9 vector PDFs in paper/figures | done |
| F5 | MiKTeX installed on this laptop; main.pdf (8 pages + references) and supp.pdf (8 pages) compile clean | done |
