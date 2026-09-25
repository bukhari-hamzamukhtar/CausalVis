"""
v7_2/train_long.py  —  fine-tune v7's learned world model on LONG simulations
==============================================================================

v7's world model (v6_voxel.pt) was trained on 20-frame rollouts. Inside a
"what if" world it runs on its own for 30-100 frames, and the shadow replay
(probe/shadow_replay.py) measured the cost: it reproduces 96.7% of real
collisions when started 12 frames before them, 55.0% when started 30 before.
This fine-tunes the SAME model on longer rollouts. Three arms, identical init,
data order, seed, learning rate, batch budget and checkpoint rule; ONLY the
rollout-length schedule differs:

    --schedule fixed:40-60                 a random length in [40, 60] every batch
    --schedule curriculum:40,50,60,70,80   move to the next length only once the
                                           val error at this one stops improving
    --schedule fixed:20-20                 control: same fine-tune at v6's length

LONG WINDOWS NEED THE OBJECTS THAT ENTER DURING THEM
---------------------------------------------------
v5's SpatialDataset keeps only objects present for the WHOLE window. Measured on
600 TRAIN clips, windows in which a kept object collides with a dropped one:
1.9% at 20 frames, 20.5% at 50, 46.7% at 80. The recording then shows a bounce
off an object the simulation does not contain, and a long-horizon run would be
trained on that. So here an object that appears mid-window is handed to the
simulator at its first frame, the way v7's world builder does it (velocity
averaged over the last 3 recorded frames, v7.1's entry correction from
v7_1/entry_profile.json, fitted on TRAIN); the loss skips an entering object's
first 15 frames, where its recorded centre is known to be biased; an object that
disappears is dropped. All three arms use these windows, so the control absorbs
any effect of the data handling itself.

CHECKPOINTS (rule fixed before any run)
---------------------------------------
Every --check-every batches the model simulates one FIXED set of val windows
(first --val-clips val clips, 80 frames, same windows for every arm). Selection
metric: mean position error over frames 1-80, lower is better; the best is
saved. Also reported: error at frames 20/40/60/80.
The curriculum reads mean error over frames 1-H of the same rollout: when
--patience checks in a row fail to beat the stage's best by --min-gain (or after
--stage-max batches) it moves to the next length; when the last length stops
improving it stops. Every arm stops at --budget batches at the latest.

    python v7_2/train_long.py --schedule curriculum:40,50,60,70,80 --out v7_2/ckpt/curriculum.pt
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v5", "v6"):
    sys.path.insert(0, os.path.join(ROOT, d))

from world import SpatialWorld, strip_colour                  # noqa: E402
from train import wrap_quarter                                # noqa: E402
from batch import resolve_voxel_batch, shapes_from_attrs      # noqa: E402

VEL_SMOOTH = 3          # v7 cf_world.to_sim
VAL_H = 80
M_MAX = 8


class LongWindows:
    """Windows of H frames in which objects that enter are handed over on arrival."""

    def __init__(self, files, entry):
        self.files = files
        self.entry = entry
        self.K = len(entry["ratio"])

    def start_state(self, pos, vel, live, k, f, first):
        """v7's to_sim velocity, plus v7.1's entry correction for a fresh entrant."""
        us = [u for u in range(max(0, f - VEL_SMOOTH + 1), f + 1) if live[u, k]]
        v0 = np.mean(vel[us, k], axis=0) if us else vel[f, k]
        v0 = np.asarray(v0, np.float64)
        p0 = pos[f, k].astype(np.float64)
        tau = f - first
        if first > 0 and 0 <= tau < self.K:
            v0 = v0 / max(self.entry["ratio"][tau], 0.3)
            p0 = p0 - self.entry["lag"][tau] * v0
        return p0.astype(np.float32), v0.astype(np.float32)

    def item(self, i, H, rng):
        z = np.load(self.files[i], allow_pickle=True)
        pos = np.nan_to_num(z["positions"].astype(np.float32))
        vel = np.nan_to_num(z["velocities"].astype(np.float32))
        pres = z["presence"]
        attrs = np.nan_to_num(z["attrs"].astype(np.float32))
        yaw = np.nan_to_num(z["yaw"].astype(np.float32)) if "yaw" in z else np.zeros(pres.shape, np.float32)
        T = pos.shape[0]
        N = min(M_MAX, pos.shape[1])
        M = M_MAX
        live = pres[:, :N] > 0
        ok = [t for t in range(1, T - H - 1) if live[t].sum() >= 2]
        t0 = int(rng.choice(ok)) if ok else 1

        alive = np.zeros((H + 1, M), np.float32)       # in the scene at frame t0+s
        inj = np.zeros((H + 1, M), np.float32)         # handed to the simulator at frame t0+s
        qin = np.zeros((H + 1, M, 2), np.float32)
        vin = np.zeros((H + 1, M, 2), np.float32)
        yin = np.zeros((H + 1, M), np.float32)
        win = np.zeros((H + 1, M), np.float32)
        lm = np.zeros((H, M), np.float32)              # loss mask for the prediction of frame t0+u+1
        for k in range(N):
            on = np.where(live[:, k])[0]
            if not on.size:
                continue
            first = int(on[0])
            if live[t0, k]:
                s0 = 0
            elif t0 < first <= t0 + H - 1:
                s0 = first - t0
            else:
                continue                                # not here yet, or it left before the window
            s = s0
            while s <= H and live[t0 + s, k]:
                alive[s, k] = 1.0
                s += 1
            f = t0 + s0
            qin[s0, k], vin[s0, k] = self.start_state(pos, vel, live, k, f, first)
            yin[s0, k] = yaw[f, k]
            win[s0, k] = wrap_quarter(yaw[f + 1, k] - yaw[f, k])
            inj[s0, k] = 1.0
            for u in range(H):
                if alive[u, k] and alive[u + 1, k] and (first == 0 or t0 + u + 1 - first >= self.K):
                    lm[u, k] = 1.0
        qf = np.zeros((H, M, 2), np.float32)
        vf = np.zeros((H, M, 2), np.float32)
        qf[:, :N] = pos[t0 + 1: t0 + 1 + H, :N]
        vf[:, :N] = vel[t0 + 1: t0 + 1 + H, :N]
        at = np.zeros((M, attrs.shape[1]), np.float32)
        at[:N] = attrs[:N]
        return at, alive, inj, qin, vin, yin, win, lm, qf, vf

    def batch(self, idxs, H, rng):
        items = [self.item(int(i), H, rng) for i in idxs]
        return [torch.from_numpy(np.stack(x)) for x in zip(*items)]


def rollout(model, at, alive, inj, qin, vin, yin, win, H, create_graph=True, stats=None):
    """v6's training rollout (voxel contacts, then the learned step), with objects
    handed over when they enter and switched off when they leave."""
    phys = strip_colour(at)
    mass, radius, e = model.properties(phys)
    I = model.inertia(phys, mass, radius)
    shapes = shapes_from_attrs(at)
    rest = at[..., 1]                                   # metal 0.8 / rubber 0.5
    B, _, M = alive.shape
    q = torch.zeros(B, M, 2)
    p = torch.zeros(B, M, 2)
    yaw = torch.zeros(B, M)
    L = torch.zeros(B, M)
    qs, vs = [], []
    for s in range(H):
        g = inj[:, s] > 0
        if bool(g.any()):
            g2 = g.unsqueeze(-1)
            q = torch.where(g2, qin[:, s], q)
            p = torch.where(g2, mass.unsqueeze(-1) * vin[:, s], p)
            yaw = torch.where(g, yin[:, s], yaw)
            L = torch.where(g, I * win[:, s], L)
            if stats is not None and s > 0:
                stats["handed_over"] += int(g.sum())
        pr = alive[:, s]
        p = resolve_voxel_batch(q, p, mass, radius, pr, rest, shapes, yaw)
        q, p, yaw, L = model.step_spatial(q, p, yaw, L, mass, radius, e, pr, phys, 1.0,
                                          create_graph)
        qs.append(q)
        vs.append(p / mass.unsqueeze(-1))
    return torch.stack(qs, 1), torch.stack(vs, 1)


def val_curve(model, vwin):
    """Mean position error at every frame 1..80 over the fixed val windows."""
    model.eval()
    tot = torch.zeros(VAL_H, dtype=torch.float64)
    cnt = torch.zeros(VAL_H, dtype=torch.float64)
    for b in vwin:
        qs, _ = rollout(model, *b[:7], VAL_H, create_graph=False)
        d = (qs.detach() - b[8]).norm(dim=-1)            # [B,H,M]
        tot += (d * b[7]).sum(dim=(0, 2)).double()
        cnt += b[7].sum(dim=(0, 2)).double()
    model.train()
    return (tot / cnt.clamp_min(1)).numpy()


def parse_schedule(s):
    kind, _, spec = s.partition(":")
    if kind == "fixed":
        lo, hi = (int(x) for x in spec.split("-"))
        return kind, (lo, hi)
    if kind == "curriculum":
        return kind, [int(x) for x in spec.split(",")]
    raise SystemExit("--schedule must be fixed:LO-HI or curriculum:H1,H2,...")


def mem():
    """Peak memory of this process so far (Windows API; blank elsewhere)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return "  peak %.2f GB" % (pmc.PeakWorkingSetSize / 1e9)
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--init", default="v6_voxel.pt")
    ap.add_argument("--data", default="data/trajectories_3d_yaw")
    ap.add_argument("--split", default="split3d_yaw.json")
    ap.add_argument("--budget", type=int, default=1500, help="max training batches, same for every arm")
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--stage-max", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-clips", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    kind, spec = parse_schedule(a.schedule)

    from splits import load_split
    have = set(os.path.basename(f) for f in glob.glob(os.path.join(a.data, "*.npz")))
    train_files = [os.path.join(a.data, os.path.basename(f))
                   for f in load_split(a.split, "train") if os.path.basename(f) in have]
    val_files = [os.path.join(a.data, os.path.basename(f))
                 for f in load_split(a.split, "val") if os.path.basename(f) in have][: a.val_clips]
    if a.limit:
        train_files = train_files[: a.limit]
    entry = json.load(open(os.path.join(ROOT, "v7_1", "entry_profile.json")))
    ds = LongWindows(train_files, entry)
    vds = LongWindows(val_files, entry)
    vrng = np.random.default_rng(12345)                  # the SAME val windows in every arm
    vwin = [vds.batch(range(i, min(i + a.batch, len(val_files))), VAL_H, vrng)
            for i in range(0, len(val_files), a.batch)]
    print(f"schedule {a.schedule}   init {a.init}   lr {a.lr}   budget {a.budget} batches of {a.batch}")
    print(f"train clips {len(train_files)}   val clips {len(val_files)} (fixed 80-frame windows)", flush=True)

    model = SpatialWorld(dissipative=True, yaw_known=False)
    model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model.freeze_radius = True                          # v5b / v6 setting: measured radius
    frozen = {id(p_) for p_ in model.radius_corr.parameters()}
    for p_ in model.radius_corr.parameters():
        p_.requires_grad_(False)
    trainable = [p_ for p_ in model.parameters() if id(p_) not in frozen]
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    B = a.batch
    order_rng = np.random.default_rng(a.seed)            # identical clip order in every arm
    win_rng = np.random.default_rng(a.seed + 1)          # window start frames
    len_rng = np.random.default_rng(a.seed + 2)          # lengths for fixed:LO-HI
    passes = math.ceil(a.budget * B / len(train_files)) + 1
    order = np.concatenate([order_rng.permutation(len(train_files)) for _ in range(passes)])
    hist_path = os.path.splitext(a.out)[0] + "_history.json"
    hist = []

    def curve_text(errs):
        return (f"mean1-80 {errs.mean():.4f}  @20 {errs[19]:.4f} @40 {errs[39]:.4f} "
                f"@60 {errs[59]:.4f} @80 {errs[79]:.4f}")

    def record(bi, H, errs, extra):
        rec = {"batch": bi, "H": H, "mean_1_80": float(errs.mean()), "at20": float(errs[19]),
               "at40": float(errs[39]), "at60": float(errs[59]), "at80": float(errs[79])}
        rec.update(extra)
        hist.append(rec)
        json.dump(hist, open(hist_path, "w"), indent=1)

    errs = val_curve(model, vwin)
    best = float(errs.mean())
    torch.save(model.state_dict(), a.out)
    stage, stage_start, bad = 0, 0, 0
    stage_best = float(errs[: spec[0]].mean()) if kind == "curriculum" else None
    record(0, None, errs, {"saved": True, "note": "init, no training"})
    print(f"check batch    0  (init)  val {curve_text(errs)}  <- saved{mem()}", flush=True)

    t_start = t_chunk = time.time()
    tot, n, bi = 0.0, 0, 0
    stats = {"handed_over": 0}
    while bi < a.budget:
        H = spec[stage] if kind == "curriculum" else int(len_rng.integers(spec[0], spec[1] + 1))
        b = ds.batch(order[bi * B:(bi + 1) * B], H, win_rng)
        qs, vs = rollout(model, *b[:7], H, stats=stats)
        m = b[7].unsqueeze(-1)
        loss = (((qs - b[8]) * m) ** 2).mean() + 0.1 * (((vs - b[9]) * m) ** 2).mean()
        bi += 1
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tot += loss.item()
            n += 1
        if bi % a.print_every == 0:
            print(f"   batch {bi:4d}/{a.budget}  H {H}  loss {tot / max(n, 1):.6f}  "
                  f"handed over mid-window {stats['handed_over']}  "
                  f"{(time.time() - t_chunk) / a.print_every:.2f}s/batch{mem()}", flush=True)
            t_chunk = time.time()
        if bi % a.check_every and bi < a.budget:
            continue

        errs = val_curve(model, vwin)
        sel = float(errs.mean())
        tag = ""
        if sel < best:
            best = sel
            torch.save(model.state_dict(), a.out)
            tag = "  <- best, saved"
        extra = {"saved": bool(tag), "train_loss": tot / max(n, 1),
                 "minutes": (time.time() - t_start) / 60}
        line = f"check batch {bi:4d}  H {H}  val {curve_text(errs)}  {extra['minutes']:.0f} min"
        stop = False
        if kind == "curriculum":
            cur = float(errs[: spec[stage]].mean())
            if cur < stage_best * (1 - a.min_gain):
                stage_best, bad = cur, 0
            else:
                bad += 1
            extra.update({"stage_H": spec[stage], "stage_err": cur, "stage_best": stage_best,
                          "checks_without_gain": bad})
            line += f"  | stage {spec[stage]}: err1-{spec[stage]} {cur:.4f} (best {stage_best:.4f}, no gain x{bad})"
            if bad >= a.patience or bi - stage_start >= a.stage_max:
                why = "stopped improving" if bad >= a.patience else "stage batch cap"
                if stage == len(spec) - 1:
                    line += f"  -> last length {why}: done"
                    extra["event"] = "done: " + why
                    stop = True
                else:
                    stage += 1
                    stage_start, bad = bi, 0
                    stage_best = float(errs[: spec[stage]].mean())
                    line += f"  -> {why}, moving to {spec[stage]} frames"
                    extra["event"] = f"advance to {spec[stage]}: {why}"
        record(bi, H, errs, extra)
        print(line + tag, flush=True)
        tot, n = 0.0, 0
        if stop:
            break
    print(f"\nsaved -> {a.out}   best val mean1-80 {best:.4f}   batches trained {bi}")


if __name__ == "__main__":
    main()
