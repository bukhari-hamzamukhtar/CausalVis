# src/data/build_dataset_v2.py
# ─────────────────────────────────────────────────────────────
# Builds causal_dataset_v2.pt
#
# Differences from V1:
#   1. 4D edge attributes per collision (distance, angle,
#      closing speed, material-weighted force proxy)
#   2. Post-collision velocity targets extracted from frames
#      [T+1, T+5] — used by Lagrange momentum constraint
#   3. Mass proxy and restitution stored per-graph for loss fn
#
# CHANGES IN THIS VERSION:
#   • exits_after() now takes `collisions` and only credits the
#     LAST collision before an exit (single-attribution fix —
#     previously one exit event inflated every prior collision
#     for that object into a positive label, regardless of how
#     many other collisions happened in between).
#   • estimate_velocity / estimate_post_velocity / build_node_features
#     now use a least-squares fit over the whole available window
#     instead of a single last-two-frame difference. Strictly
#     equivalent at exactly 2 points, more noise-robust with more.
#
# Run locally:
#   python -u "g:\CausalVis\src\data\build_dataset_v2.py"
# ─────────────────────────────────────────────────────────────

import os, sys, json
import numpy as np
import torch
from torch_geometric.data import Data
from pycocotools import mask as mask_utils

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Attribute vocabularies ────────────────────────────────────
COLORS    = ['gray','red','blue','green','brown','purple','cyan','yellow']
MATERIALS = ['rubber','metal']
SHAPES    = ['cube','sphere','cylinder']

# Physical property lookup (CLEVRER physics engine values)
MASS_PROXY  = {'rubber': 1.0, 'metal': 1.5}
RESTITUTION = {'rubber': 0.5, 'metal': 0.8}
FRICTION    = {'rubber': 0.6, 'metal': 0.3}


def one_hot(val, categories):
    v = [0.0] * len(categories)
    if val in categories:
        v[categories.index(val)] = 1.0
    return v


def get_centroid(rle):
    binary = mask_utils.decode(rle)
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def load_positions(frames):
    """
    Returns {frame_idx: {color_mat_shape_key: (cx, cy)}}
    """
    positions = {}
    for frame in frames:
        fi = frame['frame_index']
        positions[fi] = {}
        for det in frame['objects']:
            key = f"{det['color']}_{det['material']}_{det['shape']}"
            c = get_centroid(det['mask'])
            if c:
                positions[fi][key] = c
    return positions


def estimate_velocity(obj_key, frame, positions, window=5):
    """
    Velocity estimate via least-squares linear fit over the whole
    available window, not a single last-two-frame difference.

    WHY: segmentation-mask centroids jitter a little frame to frame
    (rasterization/anti-aliasing noise) even at >99% detection
    confidence — confidence measures detection quality, not sub-pixel
    centroid precision. A 2-point finite difference is fully exposed
    to whichever single frame happens to be noisy; a line fit through
    every point in the window averages that jitter out. With exactly
    2 points available this is IDENTICAL to the old finite-difference
    version — strictly better, never worse.
    """
    ts, xs, ys = [], [], []
    for f in range(max(0, frame - window), frame + 1):
        if f in positions and obj_key in positions[f]:
            ts.append(f)
            xs.append(positions[f][obj_key][0])
            ys.append(positions[f][obj_key][1])
    if len(ts) < 2:
        return 0.0, 0.0
    t_arr = np.array(ts, dtype=np.float64)
    vx = float(np.polyfit(t_arr, xs, 1)[0])
    vy = float(np.polyfit(t_arr, ys, 1)[0])
    return vx, vy


def estimate_post_velocity(obj_key, frame, positions, window=5):
    """
    Post-collision velocity via least-squares over ALL available
    frames in (frame, frame+window] — the old version used only the
    FIRST TWO post-collision points (pts[1]-pts[0]) and silently
    discarded the rest of the window even when more frames were
    available. This is the ground-truth target for the Lagrange
    momentum loss, so its noise-robustness directly affects training
    signal quality, not just cosmetic node features.
    """
    max_frame = max(positions.keys()) if positions else frame + window
    ts, xs, ys = [], [], []
    for f in range(frame + 1, min(frame + window + 1, max_frame + 1)):
        if f in positions and obj_key in positions[f]:
            ts.append(f)
            xs.append(positions[f][obj_key][0])
            ys.append(positions[f][obj_key][1])
    if len(ts) < 2:
        return 0.0, 0.0
    t_arr = np.array(ts, dtype=np.float64)
    vx = float(np.polyfit(t_arr, xs, 1)[0])
    vy = float(np.polyfit(t_arr, ys, 1)[0])
    return vx, vy


def compute_edge_features(pos0, pos1, vel0, vel1, mat0):
    """
    4D edge attribute vector for one directed edge.

    [0] Euclidean distance (normalized by frame width 480)
    [1] Impact angle     (normalized to [0,1] via /(2π))
    [2] Closing speed    (dot product of relative vel with direction)
    [3] Force proxy      (material mass × max(closing_speed, 0))
    """
    dx = pos1[0] - pos0[0]
    dy = pos1[1] - pos0[1]
    dist = max(np.sqrt(dx**2 + dy**2), 1e-6)

    angle        = np.arctan2(dy, dx)                    # [-π, π]
    angle_norm   = (angle + np.pi) / (2 * np.pi)        # → [0, 1]

    rel_vx = vel1[0] - vel0[0]
    rel_vy = vel1[1] - vel0[1]
    # Closing speed: how fast objects approach along direction vector
    closing = -(rel_vx * dx + rel_vy * dy) / dist

    mass     = MASS_PROXY[mat0]
    force    = mass * max(closing, 0.0)                  # only positive (approach)

    return [
        dist / 480.0,          # normalized distance
        angle_norm,            # normalized angle
        closing / 50.0,        # normalized closing speed
        force   / 5.0,         # normalized force proxy
    ]


def build_node_features(obj, traj):
    """
    17D node feature vector (same layout as before). Velocity is now
    the least-squares slope across the whole trajectory window
    instead of a last-two-point difference — see estimate_velocity
    for why. Assumes frames within the window are consecutive (true
    in practice given CLEVRER's consistently high detection
    confidence); if a frame's detection is ever missing, this treats
    the remaining points as evenly spaced, a reasonable approximation
    but not exact — a fully rigorous version would carry frame
    indices through `traj` itself rather than bare (x, y) tuples.
    """
    color_oh = one_hot(obj['color'],    COLORS)
    mat_oh   = one_hot(obj['material'], MATERIALS)
    shape_oh = one_hot(obj['shape'],    SHAPES)

    if len(traj) >= 2:
        t_arr = np.arange(len(traj), dtype=np.float64)
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        vx = float(np.polyfit(t_arr, xs, 1)[0])
        vy = float(np.polyfit(t_arr, ys, 1)[0])
        px, py = traj[-1]
    elif len(traj) == 1:
        vx, vy = 0.0, 0.0
        px, py = traj[0]
    else:
        vx, vy, px, py = 0.0, 0.0, 0.0, 0.0

    return color_oh + mat_oh + shape_oh + [
        px / 480.0,   # normalized x
        py / 320.0,   # normalized y
        vx,           # velocity x
        vy,           # velocity y
    ]


def exits_after(obj_id, frame, in_outs, collisions):
    """
    Single-attribution fix: True only if the object exits BEFORE its
    next collision. An object involved in 3 collisions before finally
    leaving the scene should have that exit credited to the LAST of
    the 3 — not to all 3 equally (which the old version did, since it
    only checked "does an exit happen at/after this frame", with no
    upper bound — one exit event inflated every prior collision for
    that object into a positive label).
    """
    future_collision_frames = [
        c['frame'] for c in collisions
        if obj_id in c['object'] and c['frame'] > frame
    ]
    next_collision = min(future_collision_frames) if future_collision_frames else float('inf')

    for event in in_outs:
        if (event.get('object') == obj_id and event.get('type') == 'out'
                and frame <= event.get('frame', -1) < next_collision):
            return True
    return False


def process_video_v2(json_path):
    """
    Returns list of PyG Data objects, one per collision event.

    Each Data object has:
      x          [2, 17]  node features
      edge_index [2, 2]   bidirectional edge (0→1, 1→0)
      edge_attr  [2, 4]   edge features (same for both directions)
      y          [1]      exit label (binary)
      pre_vel    [2, 2]   pre-collision (vx, vy) per object
      post_vel   [2, 2]   post-collision (vx, vy) per object
      mass       [2]      mass proxy per object
      restitution[2]      restitution coefficient per object
    """
    with open(json_path) as f:
        data = json.load(f)

    gt       = data['ground_truth']
    objects  = {o['id']: o for o in gt['objects']}
    colls    = gt['collisions']
    in_outs  = gt.get('in_outs', [])
    positions = load_positions(data['frames'])

    samples = []
    window  = 5

    for event in colls:
        frame   = event['frame']
        obj_ids = event['object']
        if len(obj_ids) != 2:
            continue

        node_feats  = []
        pre_vels    = []
        post_vels   = []
        masses      = []
        restitutions = []
        pos_at_frame = []

        for oid in obj_ids:
            obj  = objects[oid]
            key  = f"{obj['color']}_{obj['material']}_{obj['shape']}"
            mat  = obj['material']

            # Pre-collision trajectory
            traj = [positions[f][key]
                    for f in range(max(0, frame - window), frame + 1)
                    if f in positions and key in positions[f]]

            node_feats.append(build_node_features(obj, traj))
            pre_vels.append(estimate_velocity(key, frame, positions, window))
            post_vels.append(estimate_post_velocity(key, frame, positions, window))

            masses.append(MASS_PROXY[mat])
            restitutions.append(RESTITUTION[mat])

            # Position at collision frame for edge feature computation
            if frame in positions and key in positions[frame]:
                pos_at_frame.append(positions[frame][key])
            elif traj:
                pos_at_frame.append(traj[-1])
            else:
                pos_at_frame.append((0.0, 0.0))

        # ── Edge features (4D, same for both directed edges) ──
        pos0, pos1 = pos_at_frame
        vel0, vel1 = pre_vels
        ef_0to1 = compute_edge_features(
            pos0, pos1, vel0, vel1, objects[obj_ids[0]]['material'])
        ef_1to0 = compute_edge_features(
            pos1, pos0, vel1, vel0, objects[obj_ids[1]]['material'])

        # ── Exit label (single-attribution) ────────────────────
        label = 1.0 if any(
            exits_after(oid, frame, in_outs, colls) for oid in obj_ids
        ) else 0.0

        # ── Assemble PyG Data object ──────────────────────────
        x          = torch.tensor(node_feats, dtype=torch.float)   # [2, 17]
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr  = torch.tensor([ef_0to1, ef_1to0], dtype=torch.float)  # [2, 4]
        y          = torch.tensor([label], dtype=torch.float)

        pre_v  = torch.tensor(pre_vels,  dtype=torch.float)   # [2, 2]
        post_v = torch.tensor(post_vels, dtype=torch.float)   # [2, 2]
        mass   = torch.tensor(masses,    dtype=torch.float)   # [2]
        rest   = torch.tensor(restitutions, dtype=torch.float) # [2]

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        d.pre_vel     = pre_v
        d.post_vel    = post_v
        d.mass        = mass
        d.restitution = rest

        samples.append(d)

    return samples


def build_full_dataset_v2(data_dir, out_path, max_videos=None):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    if max_videos:
        files = files[:max_videos]

    print(f"Building V2 dataset from {len(files)} videos...")
    print(f"Edge attributes: distance, angle, closing speed, force proxy")
    print(f"Extra tensors:   pre_vel, post_vel, mass, restitution\n")

    all_samples = []
    pos_count   = 0

    for i, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        try:
            samples = process_video_v2(path)
            all_samples.extend(samples)
            pos_count += sum(1 for s in samples if s.y.item() == 1)
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

        if i % 100 == 0:
            print(f"  [{i:>5}/{len(files)}] "
                  f"{len(all_samples)} samples "
                  f"({pos_count} positive)")

    total = len(all_samples)
    print(f"\n{'─'*50}")
    print(f"Total samples   : {total:,}")
    print(f"Positive (exit) : {pos_count:,}  ({100*pos_count/max(total,1):.1f}%)")
    print(f"Negative (stay) : {total-pos_count:,}")
    print(f"Edge attr shape : {all_samples[0].edge_attr.shape if all_samples else 'N/A'}")
    print(f"Node feat shape : {all_samples[0].x.shape if all_samples else 'N/A'}")
    print(f"Saving to {out_path}...")

    torch.save(all_samples, out_path)
    print("Done.")
    return all_samples


if __name__ == "__main__":
    DATA_DIR = os.path.join(ROOT_DIR, 'src', 'data', 'processed_proposals')
    OUT_PATH = os.path.join(ROOT_DIR, 'data', 'causal_dataset_v2.pt')

    build_full_dataset_v2(DATA_DIR, OUT_PATH)
