"""
vlm_baseline/job.py  —  ask a vision-language model the counterfactual questions
==============================================================================

Runs on a Kaggle GPU session. For each video it pulls the mp4 straight out of CLEVRER's
remote zip (byte ranges, so about 1.6 MB per video instead of the 6.2 GB archive), samples
frames, and asks Qwen2.5-VL every counterfactual question of that video. A background thread
fetches videos a few ahead of the model, so the GPU does not wait on the network.

Two conditions, both stored:
  video   the frames are shown
  blind   the same question with no frames, which measures how much of the score comes from
          the wording of the question rather than from watching the video

    python vlm_baseline/job.py --questions questions_testA.json --out answers.jsonl \
        --frames 16 --backend hf --conditions video,blind

An answer is remembered by model, seed, prompt and frame names, and a frame name says only
where that frame sits in the clip. So a job that stops halfway can be continued, on this
machine or another one, and nothing is ever asked twice.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from remote_zip import open_zip, member_map                      # noqa: E402
from vlm import AnswerCache, Prompter, key_of                    # noqa: E402

PREAMBLE = (
    "These frames are taken in order from a short video. Objects (cubes, spheres and "
    "cylinders, in different colours, made of metal or rubber) slide on a flat surface, "
    "enter and leave the view, and collide with each other.\n"
)
BLIND_PREAMBLE = (
    "A short video shows objects (cubes, spheres and cylinders, in different colours, made "
    "of metal or rubber) sliding on a flat surface and colliding. You cannot see the video.\n"
)
TASK = (
    "\nFor each event below, say whether it happens in that situation, at any time, including "
    "after the video ends. Reply with JSON only, one entry per event, in the same order:\n"
    '{"answers": ["yes", "no", ...]}\n'
)


def build_prompt(intervention, options, blind):
    """The model is asked the physical question only. The question's polarity ("which will
    NOT happen") is applied later, when scoring, so a known weakness of language models with
    negation does not count against it here."""
    head = BLIND_PREAMBLE if blind else PREAMBLE
    lines = ["%d. %s" % (i + 1, t) for i, t in enumerate(options)]
    return (head + "\nSuppose " + intervention.rstrip(".") + ". Everything else about the "
            "scene stays the same.\nEvents:\n" + "\n".join(lines) + TASK)


def frame_keys(video, n):
    """The names an answer is remembered by: position in the clip, and nothing about the
    machine that ran the job."""
    return ["frames/video_%05d_i%02d.jpg" % (video, i) for i in range(n)]


def frame_paths(frame_dir, video, n):
    """Where those frames are actually written this session."""
    return [os.path.join(frame_dir, os.path.basename(k)).replace("\\", "/")
            for k in frame_keys(video, n)]


def sample_frames(mp4_bytes, paths, frame_dir):
    """Write len(paths) frames spread over the clip. Returns the paths actually written."""
    import cv2
    os.makedirs(frame_dir, exist_ok=True)
    tmp = os.path.join(frame_dir, "_clip.mp4")
    with open(tmp, "wb") as fh:
        fh.write(mp4_bytes)
    cap = cv2.VideoCapture(tmp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 128
    n = len(paths)
    want = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)]
    written = []
    for p, idx in zip(paths, want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(p, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written.append(p)
    cap.release()
    os.remove(tmp)
    return written


def parse_answers(text, n):
    """Pull n yes/no answers out of the reply. Unreadable ones become 'no' (the common label)."""
    import re
    out = None
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                for k in ("answers", "answer", "results"):
                    if isinstance(data.get(k), list):
                        out = data[k]
                        break
        except Exception:
            out = None
    if out is None:
        out = re.findall(r"\b(yes|no)\b", (text or "").lower())
    vals = ["yes" if str(v).strip().lower().startswith("y") else "no" for v in out[:n]]
    parsed_ok = len(vals) == n
    vals += ["no"] * (n - len(vals))
    return vals, parsed_ok


class Fetcher:
    """Reads the videos in order on a background thread, a few ahead of the model.

    A video that cannot be read is handed on as the error it raised, so one bad member does
    not stop the job. At most `ahead` clips are held, about 1.6 MB each.
    """

    DONE = "done"

    def __init__(self, videos, ahead=3):
        self.videos = list(videos)
        self.q = queue.Queue(maxsize=max(1, ahead))
        self.held = {}
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        try:
            zf = open_zip()
            mm = member_map(zf)
            print("remote zip ready: %d videos" % len(mm), flush=True)
        except Exception as e:                      # no network: every video reports the same
            for v in self.videos:
                self.q.put((v, e))
            self.q.put(self.DONE)
            return
        for v in self.videos:
            try:
                self.q.put((v, zf.read(mm[v])))
            except Exception as e:
                self.q.put((v, e))
        self.q.put(self.DONE)

    def get(self, video):
        """The bytes of one video. Clips that arrive early are held until they are asked for."""
        while video not in self.held:
            item = self.q.get()
            if item == self.DONE:
                raise KeyError("video %s was never fetched" % video)
            self.held[item[0]] = item[1]
        data = self.held.pop(video)
        if isinstance(data, Exception):
            raise data
        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(HERE, "questions_testA.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "answers.jsonl"))
    ap.add_argument("--cache", default="", help="extra answer files to read, comma separated")
    ap.add_argument("--frame-dir", default="frames")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--backend", default="hf", choices=["hf", "replay"])
    ap.add_argument("--dtype", default="mixed")
    ap.add_argument("--conditions", default="video,blind")
    ap.add_argument("--limit", type=int, default=None, help="videos, for a smoke test")
    ap.add_argument("--hours", type=float, default=11.0, help="stop before the session ends")
    ap.add_argument("--seed", type=int, default=0, help="generation seed, part of the answer key")
    ap.add_argument("--ahead", type=int, default=3, help="videos fetched ahead of the model")
    a = ap.parse_args()

    conds = [c for c in a.conditions.split(",") if c]
    qs = json.load(open(a.questions, encoding="utf-8"))
    by_video = {}
    for q in qs:
        by_video.setdefault(q["video"], []).append(q)
    videos = sorted(by_video)[:a.limit]

    cache = AnswerCache([f for f in ([a.out] + a.cache.split(",")) if f], write_file=a.out)
    pr = Prompter(model=a.model, seed=a.seed, cache=cache, backend=a.backend, dtype=a.dtype)

    # what is left to ask, worked out before anything is downloaded
    plan, skipped = {}, 0
    for v in videos:
        todo = []
        for q in by_video[v]:
            for cond in conds:
                prompt = build_prompt(q.get("intervention") or q["question"], q["options"],
                                      cond == "blind")
                names = [] if cond == "blind" else frame_keys(v, a.frames)
                if cache.get(key_of(pr.model_name, pr.seed, prompt, names)) is None:
                    todo.append((q, cond, prompt, names))
                else:
                    skipped += 1
        if todo:
            plan[v] = todo
    order = [v for v in videos if v in plan]
    need_frames = [v for v in order if any(c == "video" for _, c, _, _ in plan[v])]
    print("%d videos with work | %d questions to ask, %d already answered | %d videos to fetch"
          % (len(order), sum(len(t) for t in plan.values()), skipped, len(need_frames)),
          flush=True)
    if not order:
        return
    fetch = Fetcher(need_frames, ahead=a.ahead) if need_frames else None

    t_start = time.time()
    asked = failed = 0
    for n, v in enumerate(order):
        if time.time() - t_start > a.hours * 3600:
            print("time budget reached, stopping cleanly", flush=True)
            break
        todo = plan[v]
        paths = frame_paths(a.frame_dir, v, a.frames)
        if any(c == "video" for _, c, _, _ in todo):
            try:
                sample_frames(fetch.get(v), paths, a.frame_dir)
            except Exception as e:                  # a video we cannot read must not end the job
                print("video %d unavailable: %s" % (v, e), flush=True)
                todo = [t for t in todo if t[1] != "video"]
        for q, cond, prompt, names in todo:
            use = [] if cond == "blind" else paths
            try:
                pr.ask(prompt, use, max_new_tokens=128, key_paths=names)
                asked += 1
            except Exception as e:
                failed += 1
                print("question %s/%s %s failed: %s" % (v, q["qid"], cond, e), flush=True)
        for p in paths:                             # keep the session's disk small
            try:
                os.remove(p)
            except OSError:
                pass
        if (n + 1) % 10 == 0 or n + 1 == len(order):
            el = (time.time() - t_start) / 60
            rate = pr.gpu_seconds / max(1, pr.calls)
            ahead = (len(order) - n - 1) * (el / (n + 1))
            print("%d/%d videos | asked %d, failed %d | %.1f min used, about %.0f min left | "
                  "%.1f s per question" % (n + 1, len(order), asked, failed, el, ahead, rate),
                  flush=True)
    print("done: %d new answers, %.1f GPU minutes, file %s" %
          (pr.calls, pr.gpu_seconds / 60, a.out), flush=True)


if __name__ == "__main__":
    main()
