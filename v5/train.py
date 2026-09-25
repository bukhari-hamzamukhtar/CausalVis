"""
v5/train.py  —  train the orientation-aware world model on yaw-fitted 3D data
=============================================================================

ONE conceptual change against v3_6_3d: the model can see orientation. That
brings three inseparable parts (you cannot have torque without a yaw-dependent
potential, and you cannot supervise yaw without a yaw loss):
  - shape-aware contact through support functions      (world.V)
  - rotational state and torque by autograd            (world.step_spatial)
  - a yaw loss that respects a cube's 4-fold symmetry   (below)

Everything else is held at the v3_6_3d setting: legacy loss normalisation,
learnable radius correction, drag on, same curriculum, same optimiser.

FAIR COMPARISON
---------------
The yaw-fitted data arrives in file order, so early on only the first few
thousand clips exist. Run this script TWICE on the same subset:
    --no-yaw   (yaw_known=False -> reduces exactly to v3_6's physics)
    --yaw      (yaw_known=True)
Same clips, same epochs, one variable. Compare on the test clips of that same
subset. Do not compare a 3k-clip model against the 16k-clip v3_6_3d.

    python v5/train.py --data data/trajectories_3d_yaw --split split3d.json \
        --yaw --epochs 30 --out v5_spatial.pt
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src2"))
from world import SpatialWorld, strip_colour                 # noqa: E402
from dynamics import RADIUS_COL_RAW                          # noqa: E402

QUARTER = math.pi / 2          # a cube's yaw is defined modulo 90 degrees


def wrap_quarter(a):
    """Map an angle difference into (-pi/4, pi/4] -- 4-fold symmetry."""
    return (a + QUARTER / 2) % QUARTER - QUARTER / 2


class SpatialDataset(torch.utils.data.Dataset):
    def __init__(self, files, horizon=3, max_objects=8):
        self.files = files
        self.horizon = horizon
        self.M = max_objects

    def __len__(self):
        return len(self.files)

    def set_horizon(self, h):
        self.horizon = h

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        pos, vel, pres, attrs = z["positions"], z["velocities"], z["presence"], z["attrs"]
        yaw = z["yaw"] if "yaw" in z else np.zeros(pres.shape, np.float32)
        T, N, _ = pos.shape
        H = self.horizon
        ok = [t for t in range(1, T - H - 1)
              if (pres[t:t + H + 1].min(axis=0) > 0).sum() >= 2]
        t0 = int(np.random.choice(ok)) if ok else 1
        keep = pres[t0:t0 + H + 1].min(axis=0) > 0
        idx = np.where(keep)[0][: self.M]
        M = self.M

        def pad(a, last):
            out = np.zeros(a.shape[:-1] + (M,) + last, np.float32) if last else \
                np.zeros(a.shape[:-1] + (M,), np.float32)
            out[..., : len(idx)] = a[..., idx] if not last else a[..., idx, :]
            return out

        q0 = np.zeros((M, 2), np.float32); q0[: len(idx)] = pos[t0, idx]
        v0 = np.zeros((M, 2), np.float32); v0[: len(idx)] = vel[t0, idx]
        y0 = np.zeros((M,), np.float32); y0[: len(idx)] = yaw[t0, idx]
        # angular velocity at t0 from the wrapped yaw difference
        w0 = np.zeros((M,), np.float32)
        w0[: len(idx)] = wrap_quarter(yaw[t0 + 1, idx] - yaw[t0, idx])
        at = np.zeros((M, attrs.shape[1]), np.float32); at[: len(idx)] = attrs[idx]
        pr = np.zeros((M,), np.float32); pr[: len(idx)] = 1.0
        qf = np.zeros((H, M, 2), np.float32); qf[:, : len(idx)] = pos[t0 + 1: t0 + 1 + H, idx]
        vf = np.zeros((H, M, 2), np.float32); vf[:, : len(idx)] = vel[t0 + 1: t0 + 1 + H, idx]
        yf = np.zeros((H, M), np.float32); yf[:, : len(idx)] = yaw[t0 + 1: t0 + 1 + H, idx]
        return tuple(torch.from_numpy(x) for x in (q0, v0, y0, w0, at, pr, qf, vf, yf))


def val_error(model, dl, H):
    model.eval()
    tot, n = 0.0, 0
    for q0, v0, y0, w0, at, pr, qf, vf, yf in dl:
        qs, _, _, _ = model.rollout_spatial(q0, v0, y0, w0, at, pr, H, create_graph=False)
        err = (qs[:, -1] - qf[:, -1]).norm(dim=-1)
        live = pr > 0
        tot += float(err[live].sum().detach()); n += int(live.sum())
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories_3d_yaw")
    ap.add_argument("--split", default="split3d.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--yaw", dest="yaw", action="store_true", default=False)
    ap.add_argument("--no-yaw", dest="yaw", action="store_false")
    ap.add_argument("--yaw-weight", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the clip count (use the SAME cap for both runs)")
    ap.add_argument("--max-clip", type=int, default=None,
                    help="only clips numbered below this. FREEZES the set while the "
                         "yaw extractor is still adding files, so --no-yaw and --yaw "
                         "train on identical clips")
    ap.add_argument("--freeze-radius", action="store_true",
                    help="lock radius at the measured value. Without this the "
                         "correction collapses to 0.7000 and the cube half-side "
                         "derived from it is 30%% too small -- which confounds "
                         "any orientation experiment")
    ap.add_argument("--val-clips", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from splits import load_split
    have = set(os.path.basename(f) for f in glob.glob(os.path.join(a.data, "*.npz")))
    if a.max_clip is not None:
        import re
        have = set(f for f in have if int(re.search(r"(\d+)", f).group(1)) < a.max_clip)
    train_files = [f for f in load_split(a.split, "train") if os.path.basename(f) in have]
    val_files = [f for f in load_split(a.split, "val") if os.path.basename(f) in have]
    # split.json's data_dir points elsewhere; re-root onto this folder
    train_files = [os.path.join(a.data, os.path.basename(f)) for f in train_files]
    val_files = [os.path.join(a.data, os.path.basename(f)) for f in val_files][: a.val_clips]
    if a.limit:
        train_files = train_files[: a.limit]
    print(f"yaw_known={a.yaw}  train clips {len(train_files)}  val clips {len(val_files)}")
    if not train_files:
        raise SystemExit("no training clips with yaw yet -- wait for the extraction")

    ds = SpatialDataset(train_files)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True)
    vds = SpatialDataset(val_files, horizon=20)
    vdl = torch.utils.data.DataLoader(vds, batch_size=a.batch, shuffle=False)

    model = SpatialWorld(dissipative=True, yaw_known=a.yaw)
    trainable = list(model.parameters())
    if a.freeze_radius:
        model.freeze_radius = True
        for p_ in model.radius_corr.parameters():
            torch.nn.init.zeros_(p_); p_.requires_grad_(False)
        frozen = {id(p_) for p_ in model.radius_corr.parameters()}
        trainable = [p_ for p_ in model.parameters() if id(p_) not in frozen]
        print("radius: FROZEN at the measured value")
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    best = float("inf")
    for ep in range(a.epochs):
        H = min(3 + ep, 20)
        ds.set_horizon(H)
        tot, n = 0.0, 0
        for q0, v0, y0, w0, at, pr, qf, vf, yf in dl:
            qs, vs, ys, ws = model.rollout_spatial(q0, v0, y0, w0, at, pr, H)
            m = pr[:, None, :, None]
            # v3_6's exact objective, plus a yaw term only when yaw is known
            loss = (((qs - qf) * m) ** 2).mean() + 0.1 * (((vs - vf) * m) ** 2).mean()
            if a.yaw:
                cube = at[:, None, :, 2] * pr[:, None, :]              # [B,1,M]
                dy = ys - yf
                loss = loss + a.yaw_weight * ((1 - torch.cos(4 * dy)) * cube).mean()
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); tot += loss.item(); n += 1

        with torch.no_grad():
            phys = strip_colour(at)
            mass, radius, e = model.properties(phys)
            rr = (radius[pr > 0] / at[..., RADIUS_COL_RAW][pr > 0].clamp_min(1e-9)).mean().item()
        ve = val_error(model, vdl, 20)
        tag = ""
        if ve < best:
            best = ve; torch.save(model.state_dict(), a.out); tag = "  <- best, saved"
        print(f"epoch {ep:3d}  horizon {H:2d}  loss {tot/max(n,1):.6f}  "
              f"r/r_meas {rr:.4f}  val_err@20 {ve:.4f}{tag}", flush=True)
    print(f"\nsaved -> {a.out}   best val err@20 {best:.4f}")


if __name__ == "__main__":
    main()
