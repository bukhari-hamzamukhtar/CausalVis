# Per-option records

`test_records.zip` holds one JSON line per answered option for every system in the paper,
on TEST-A and TEST-B. Unzip it in the repository root so the folders land at
`paper_exp/runs/<SET>_<system>/shard*.jsonl`.

Each line has the video, question id, choice text, the truth label, the objects the question
asks about, the removed object, when each object was handed to the simulator, and the collision
frames every detector found. Everything in the paper is computed from these files:

    python paper_exp/score.py one paper_exp/runs/B_learned union 30     # one system
    python paper_exp/final_report.py                                    # the full report

Folder names: `A_` is TEST-A (500 CLEVRER validation videos), `B_` is TEST-B (898 CLEVRER
training videos never used for training). `learned` is the world model, `laws` the fitted
textbook laws, `straight` and `strfric` the straight-line simulators, `nosim` no physics,
`abl_*` the one-change ablations, `ckpt_*` other checkpoints, `seed1` and `seed2` retrains.
