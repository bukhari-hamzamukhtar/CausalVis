# Reproducing every number in the paper

All commands run from the repository root on a CPU (the paper's numbers came from a
4-core laptop, i5-8350U, 8 GB RAM). Python 3, PyTorch (CPU), NumPy.

## 1. Inputs
- CLEVRER questions (all four types, train + validation): `zechennlp/` (Hugging Face
  copy of CLEVRER's question files).
- NS-DR's released detections for all 20,000 videos: `legacy/v1/data/processed_proposals/`.
- Camera recovered from silhouettes: `camera_fit.json`.
- Video split (fixed once, before any experiment): `split3d_yaw.json`.

## 2. Tracks from detections only
    python paper_exp/detonly.py report --which train --limit 200      # tracker vs annotation
    bash paper_exp/run_build.sh                                         # -> data/trajectories_3d_det
    python paper_exp/perception_parity.py                               # same quality on every set?

## 3. Question parser (TRAIN questions only)
    python paper_exp/qparser.py fit
    python paper_exp/qparser.py check --qset val

## 4. One builder for every experiment
    python paper_exp/make_final.py        # generates paper_exp/cf_world.py and evaluate.py from v7_3

Knobs: SIM (learned | laws | straight | straight_friction), CF_LOOKBACK, NO_VOXEL,
NO_FORCE, DRAG_SCALE, PARSED; flags --model, --no-entry-fix, --no-sim.

## 5. Selection on VAL-A, then the pre-registered TEST runs
    python paper_exp/runq.py paper_exp/jobs_val_det.txt --parallel 1
    python paper_exp/make_test_jobs.py          # applies the VAL choices -> jobs_test.txt
    python paper_exp/runq.py paper_exp/jobs_test.txt --parallel 2
    python paper_exp/score.py one  paper_exp/runs/B_learned union 30
    python paper_exp/score.py pair paper_exp/runs/B_learned union 30 paper_exp/runs/B_laws union 30

## 6. Descriptive, explanatory, predictive
    PARSED=1 python paper_exp/eval_qtypes.py --which val --qset val --out paper_exp/qruns/val_A
    python paper_exp/score_qtypes.py grid paper_exp/qruns/val_A --parsed
    python paper_exp/score_qtypes.py one  paper_exp/qruns/test_B paper_exp/qruns/val_A/chosen_settings.json --parsed

## 7. Physics quality of every simulator (audit x-axis)
    python paper_exp/physics_quality.py --sim learned --model v6_voxel.pt --out paper_exp/quality/learned.json

## 8. Other benchmarks
    python paper_exp/humans.py ground --split valid     # needs GROQ_API_KEY (free tier)
    python paper_exp/humans.py score --split valid
    python paper_exp/comphy.py fit
    python paper_exp/comphy.py eval --split val --out paper_exp/comphy_runs/val.json
