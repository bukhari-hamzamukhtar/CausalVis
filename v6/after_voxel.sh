#!/usr/bin/env bash
# Runs after v6/run_voxel.sh finishes (training + three benchmarks).
# Needs GROQ_API_KEY in the environment; the key is never written to disk.
cd "G:/CausalVis"
while [ ! -f v6/VOXEL_TRAIN_DONE ]; do sleep 60; done
# the language model answers test questions from the stored future of the voxel-trained model
python -u v6/lm_eval.py --model v6_voxel.pt --limit 60 --provider groq --out v6/cmp/lm_eval_v6_voxel.jsonl > v6/cmp/lm_eval_v6_voxel.log 2>&1
# the one GIF: LM parses, voxel-trained world model simulates, store, LM answers, orbiting render
python -u v6/demo.py --model v6_voxel.pt --index 5 --provider groq --out v6/demo_trained > v6/demo_trained.log 2>&1
echo DONE > v6/AFTER_DONE
