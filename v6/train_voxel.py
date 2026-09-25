"""
v6/train_voxel.py  —  train the world model WITH voxel contacts inside the loop
==============================================================================

Until now the voxel contact resolver ran only at evaluation, bolted onto a
model that had learned its potential without it. That model spent its whole
training imitating bounces with a soft spring, so switching the true contact
on afterwards gave it a mechanism it had never learned to work alongside.

Here the voxels are part of every training rollout:

    for each step:
        1. every pair's occupied voxels are checked    (v6/batch.py)
        2. pressed pairs get the rigid-body impulse along the voxel normal,
           split into along-normal (exchanged) and across-normal (kept)
        3. then the learned potential + drag advance the state (kick-drift-kick)

The impulse magnitude is differentiable in the momentum, so the loss teaches
the potential, drag and mass heads to produce approaches that, resolved by true
contact geometry, land where the recorded objects landed.

ONE VARIABLE
------------
Warm-started from the production checkpoint and fine-tuned. Run twice with the
same init, data, seed, epochs and horizon:
    --voxel      contacts from voxel occupancy inside training
    --no-voxel   identical fine-tune without them (the control)
Otherwise any gain could just be "trained for longer".

    python v6/train_voxel.py --init v5b_noyaw.pt --voxel    --out v6_voxel.pt
    python v6/train_voxel.py --init v5b_noyaw.pt --no-voxel --out v6_control.pt
"""

import argparse
import glob
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
from train import SpatialDataset                              # noqa: E402
from batch import resolve_voxel_batch, shapes_from_attrs      # noqa: E402


def rollout(model, q0, v0, y0, w0, attrs, present, steps, voxel, create_graph=True,
            stats=None):
    """rollout_spatial, with voxel contact resolution before every step.
    With voxel=False this is exactly SpatialWorld.rollout_spatial."""
    phys = strip_colour(torch.nan_to_num(attrs))
    mass, radius, e = model.properties(phys)
    I = model.inertia(phys, mass, radius)
    q = torch.nan_to_num(q0) * present.unsqueeze(-1)
    p = mass.unsqueeze(-1) * torch.nan_to_num(v0) * present.unsqueeze(-1)
    yaw = torch.nan_to_num(y0) * present
    L = I * torch.nan_to_num(w0) * present
    shapes = shapes_from_attrs(attrs) if voxel else None
    rest = attrs[..., 1]                          # metal 0.8 / rubber 0.5
    qs, vs = [], []
    for _ in range(steps):
        if voxel:
            ev = [] if stats is not None else None
            p = resolve_voxel_batch(q, p, mass, radius, present, rest, shapes, yaw,
                                    collect=ev)
            if stats is not None:
                stats["contacts"] += len(ev)
        q, p, yaw, L = model.step_spatial(q, p, yaw, L, mass, radius, e, present,
                                          phys, 1.0, create_graph)
        qs.append(q); vs.append(p / mass.unsqueeze(-1))
    return torch.stack(qs, 1), torch.stack(vs, 1)


def val_error(model, dl, H, voxel):
    model.eval()
    tot, n = 0.0, 0
    for q0, v0, y0, w0, at, pr, qf, vf, yf in dl:
        qs, _ = rollout(model, q0, v0, y0, w0, at, pr, H, voxel, create_graph=False)
        err = (qs[:, -1] - qf[:, -1]).norm(dim=-1)
        live = pr > 0
        tot += float(err[live].sum().detach()); n += int(live.sum())
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories_3d_yaw")
    ap.add_argument("--split", default="split3d_yaw.json")
    ap.add_argument("--init", default="v5b_noyaw.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voxel", dest="voxel", action="store_true", default=True)
    ap.add_argument("--no-voxel", dest="voxel", action="store_false")
    ap.add_argument("--val-clips", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    from splits import load_split
    have = set(os.path.basename(f) for f in glob.glob(os.path.join(a.data, "*.npz")))
    train_files = [os.path.join(a.data, os.path.basename(f))
                   for f in load_split(a.split, "train") if os.path.basename(f) in have]
    val_files = [os.path.join(a.data, os.path.basename(f))
                 for f in load_split(a.split, "val") if os.path.basename(f) in have][: a.val_clips]
    if a.limit:
        train_files = train_files[: a.limit]
    print(f"VOXEL CONTACTS IN TRAINING = {a.voxel}   init {a.init}")
    print(f"train clips {len(train_files)}  val clips {len(val_files)}  horizon {a.horizon}")

    ds = SpatialDataset(train_files, horizon=a.horizon)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                     generator=g)
    vds = SpatialDataset(val_files, horizon=20)
    vdl = torch.utils.data.DataLoader(vds, batch_size=a.batch, shuffle=False)

    model = SpatialWorld(dissipative=True, yaw_known=False)
    model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model.freeze_radius = True                       # v5b's setting: measured radius
    frozen = {id(p_) for p_ in model.radius_corr.parameters()}
    for p_ in model.radius_corr.parameters():
        p_.requires_grad_(False)
    trainable = [p_ for p_ in model.parameters() if id(p_) not in frozen]
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    np.random.seed(a.seed + 1)                       # same val windows both arms
    best = val_error(model, vdl, 20, a.voxel)
    torch.save(model.state_dict(), a.out)
    print(f"epoch  -1  (init, no training)  val_err@20 {best:.4f}  <- saved", flush=True)

    for ep in range(a.epochs):
        np.random.seed(a.seed + 100 + ep)
        tot, n, t0 = 0.0, 0, time.time()
        stats = {"contacts": 0}
        for bi, (q0, v0, y0, w0, at, pr, qf, vf, yf) in enumerate(dl):
            qs, vs = rollout(model, q0, v0, y0, w0, at, pr, a.horizon, a.voxel,
                             stats=stats)
            m = pr[:, None, :, None]
            loss = (((qs - qf) * m) ** 2).mean() + 0.1 * (((vs - vf) * m) ** 2).mean()
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); tot += loss.item(); n += 1
            if bi % 100 == 0:
                print(f"   ep {ep} batch {bi:4d}/{len(dl)}  loss {tot/max(n,1):.6f}  "
                      f"voxel contacts so far {stats['contacts']}  "
                      f"{(time.time()-t0)/(bi+1):.2f}s/batch", flush=True)
        np.random.seed(a.seed + 1)
        ve = val_error(model, vdl, 20, a.voxel)
        tag = ""
        if ve < best:
            best = ve; torch.save(model.state_dict(), a.out); tag = "  <- best, saved"
        print(f"epoch {ep:3d}  loss {tot/max(n,1):.6f}  contacts {stats['contacts']}  "
              f"val_err@20 {ve:.4f}  {(time.time()-t0)/60:.1f} min{tag}", flush=True)
    print(f"\nsaved -> {a.out}   best val err@20 {best:.4f}")


if __name__ == "__main__":
    main()
