# VLM baseline

What a general vision-language model scores on the same counterfactual options the engine is
scored on. The paper compares against other published systems and against two trivial
policies; this fills the gap in between, a model that sees the video and answers from what it
sees, with no physics and no simulation.

Model: Qwen2.5-VL-7B-Instruct, 16 frames per clip, greedy decoding, seed 0. The prompter is
the one from the OCVLP course project, which follows VLP (Wuest et al., arXiv:2511.18964):
text first, then images, at most 224x224 pixels each.

Two conditions are always run together:

| condition | what the model gets | what it measures |
|-----------|--------------------|------------------|
| `video`   | 16 frames and the question | the baseline |
| `blind`   | the question only  | how much of the score comes from the wording |

The model is asked the physical question ("does this event happen if the cube is removed?").
The question's polarity is applied when scoring, so the number is not a test of how a language
model handles the word "not".

## Files

| file | what it does |
|------|--------------|
| `make_questions.py` | writes the question set and, separately, the answer key |
| `remote_zip.py` | reads single videos out of CLEVRER's 6.2 GB zip by byte range |
| `vlm.py` | the model behind an append-only answer cache |
| `job.py` | the GPU job: fetch, sample frames, ask, store |
| `score.py` | per option and per question, bootstrap intervals, paired test against the engine |
| `selftest.py` | scores answers whose result is known in advance, no GPU needed |
| `prepare.py` | builds the Kaggle dataset and one kernel per job |
| `kernel_run.py` | the kernel template `prepare.py` fills in |
| `run_kaggle.sh` | push when a GPU is free, wait, download the answers |

## Steps

```bash
python vlm_baseline/make_questions.py --which test --out vlm_baseline/questions_testA.json
python vlm_baseline/make_questions.py --which val  --out vlm_baseline/questions_valA.json
python vlm_baseline/selftest.py                      # checks the scorer, no GPU
python vlm_baseline/prepare.py --user bukharihamzamukhtar
```

Then, once, create the code dataset (0.8 MB; the videos are fetched inside the session, so
nothing large is uploaded):

```bash
kaggle datasets create -p vlm_baseline/kaggle/build/code --dir-mode zip
```

Later builds update it instead:

```bash
kaggle datasets version -p vlm_baseline/kaggle/build/code -m "update" --dir-mode zip
```

Run the smoke job first (12 validation videos, about an hour), read its log, then the full
test job:

```bash
KAGGLE_API_TOKEN=... bash vlm_baseline/run_kaggle.sh smoke
KAGGLE_API_TOKEN=... bash vlm_baseline/run_kaggle.sh test
```

Score what came back:

```bash
python vlm_baseline/score.py table questions_testA.json vlm_baseline/kaggle/outputs/test/out/test/answers.jsonl
python vlm_baseline/score.py pair  questions_testA.json vlm_baseline/kaggle/outputs/test/out/test/answers.jsonl video A_learned cal 30
```

## If a session runs out of time

The job stops cleanly at its `--hours` budget and every answer is already on disk. Upload the
downloaded `answers.jsonl` as a dataset, then build the kernels again with
`--resume-from <that slug>` and push the same job: answers already in a cache are never asked
again.

## Numbers to compare against

On the same 3,332 test options (test set A), per option and per question:

| system                                | options | questions |
|---------------------------------------|--------:|----------:|
| never predict a collision             | 54.1%   | 0.0%      |
| always predict a collision            | 45.9%   | 2.6%      |
| delete the object, keep the recording | 81.0%   | 48.5%     |
| engine, learned world model           | 89.9%   | 70.8%     |
| engine, textbook laws                 | 91.2%   | 74.3%     |

(The paper's headline 90.4 / 91.4 is the same engine on test A and B together.)

## Result, run once on 2026-09-27

| condition | per option | per question | says "yes" |
|-----------|-----------:|-------------:|-----------:|
| 16 frames | 54.4% (1811/3332) | 2.9% (27/930) | 9% |
| blind     | 48.8% (1625/3332) | 4.2% (39/930) | 78% |

With the frames the model matches "never predict a collision" (54.4 against 54.1). It answers
"no" to every event of 86% of questions when it sees frames and "yes" to every event of 70%
when it cannot, so it is reacting to whether images are present rather than to the
intervention. Paired against the engine on the same options: engine right where the model is
wrong 1,362 times, the reverse 177, z = -30.2. Two replies of 1,859 could not be read; 4.4 s
per answer on one T4, about two hours for the whole test set.
