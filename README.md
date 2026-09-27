# CausalVis

CausalVis answers "what if" questions about videos of colliding objects. It removes or changes
an object, rebuilds the scene without it, and reads the answer off the rebuilt world. Every
answer carries the event that decided it: which two objects collide, at which frame, and whether
that event was recorded in the video or simulated.

Objects follow the recorded video until the change can reach them. At that moment they are handed
to a simulator, which can be a small learned world model, textbook laws fitted on training videos,
or straight-line motion. No part of the system is trained on questions or answers.

The paper, the figures and the per-option records of every experiment are in this repository.

## Results on CLEVRER counterfactual questions

TEST A+B: 1,398 held-out videos, 2,619 questions, 9,350 options. Settings were chosen on a
separate validation split and each test run was made once. Intervals resample videos.

| simulator inside the engine | options correct | questions correct |
|---|---|---|
| textbook laws fitted on training videos | 91.4 [90.8, 92.1] | 75.0 [73.2, 76.8] |
| learned world model (43,460 parameters) | 90.4 [89.7, 91.1] | 72.3 [70.5, 74.2] |
| straight lines with friction | 90.2 | 71.9 |
| straight lines | 89.7 | 70.2 |
| no physics (delete the object, keep the recording) | 81.4 | 50.1 |

Across 13 simulators a 2.2x spread in physics error moves the score by 2.2 points, and on
predictive questions the ranking reverses. Of the correct answers whose deciding event can be
checked against CLEVRER's annotation, 98.2% rest on a real event. Full results, ablations and the
other benchmarks (CLEVRER-Humans, ComPhy, CoPhy) are in `paper_exp/RESULTS.md`.

## Hardware

Everything, including training, ran on one laptop CPU (Intel i5-8350U, 4 cores, 7.9 GB RAM).
No GPU is needed. Training the world model took 8.1 hours; answering one question takes
2.2 seconds with the learned model and 1.4 with the textbook laws.

## Setup

Python 3.10 or newer, PyTorch (CPU build), NumPy, Pillow.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pillow

## What is where

| path | contents |
|---|---|
| `paper/` | the paper: `main.tex`, `supp.tex`, `refs.bib`, the nine figures, and the compiled PDFs |
| `paper_exp/` | everything behind the paper: engine, evaluation, scoring, audits, figure scripts |
| `paper_exp/REPRODUCE.md` | every command, in order, that produces every number |
| `paper_exp/PREREGISTRATION.md` | the test plan, written before the test runs |
| `paper_exp/RESULTS.md` | all results, including the negative ones |
| `paper_exp/records/` | per-option records of every test run, zipped |
| `app/` | interactive page: ask a what-if question in plain English, turn the 3D view, save a GIF |
| `src2/` | world model, training, renderer, diagnostics |
| `v5/` to `v8/` | the versions of the engine, in order, each with its own README notes |
| `probe/` | measurement scripts (shadow replay, fitted laws, entry-state study) |
| `legacy/`, `src/` | the first two generations of the project, kept for the record |
| `WORKING_LOG.md` | the working log: every experiment, including the ones that failed |

Model checkpoints sit in the repository root. `v6_voxel.pt` is the one used in the paper.
`v8/laws.json` holds the fitted textbook laws, `camera_fit.json` the camera recovered from object
silhouettes, and `split3d_yaw.json` the fixed train, validation and test split of the videos.

## Data (not included, download separately)

- CLEVRER videos, questions and annotations: http://clevrer.csail.mit.edu/
- The object detections we use are the ones released with NS-DR, already in
  `legacy/v1/data/processed_proposals/`.
- CLEVRER-Humans, ComPhy and CoPhy: see `paper_exp/REPRODUCE.md`, section 8.

Object tracks are built from the detections with `paper_exp/detonly.py`, which writes
`data/trajectories_3d_det/`. That folder is large and stays out of the repository.

## Run the engine on one question

    PARSED=1 CF_LOOKBACK=3 python paper_exp/evaluate.py --data data/trajectories_3d_det \
        --which test --questions zechennlp/counterfactual/validation-00000-of-00001.json \
        --limit 1 --out one_question

## Run the interactive app

    python app/server.py --port 8000

Then open http://127.0.0.1:8000. Type a change in plain English ("what if the cube were twice as
heavy", "remove the red sphere"), and the page shows the recorded video and the rebuilt world side
by side, both turnable in 3D.

## Citation

Paper in preparation; a preprint will be linked here.
