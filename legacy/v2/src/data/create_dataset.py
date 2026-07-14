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
    Returns (vx, vy) as mean delta over frames [frame-window, frame].
    Returns (0, 0) if insufficient data.
    """
    pts = []
    for f in range(max(0, frame - window), frame + 1):
        if f in positions and obj_key in positions[f]:
            pts.append(positions[f][obj_key])
    if len(pts) < 2:
        return 0.0, 0.0
    vx = pts[-1][0] - pts[-2][0]
    vy = pts[-1][1] - pts[-2][1]
    return vx, vy


def estimate_post_velocity(obj_key, frame, positions, window=5):
    """
    Returns (vx, vy) from frames [frame+1, frame+window].
    This is the velocity AFTER the collision — used as the
    ground truth target for the Lagrange momentum constraint.
    """
    pts = []
    max_frame = max(positions.keys()) if positions else frame + window
    for f in range(frame + 1, min(frame + window + 1, max_frame + 1)):
        if f in positions and obj_key in positions[f]:
            pts.append(positions[f][obj_key])
    if len(pts) < 2:
        return 0.0, 0.0
    vx = pts[1][0] - pts[0][0]
    vy = pts[1][1] - pts[0][1]
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
    17D node feature vector (same as V1 for backward compatibility).
    """
    color_oh = one_hot(obj['color'],    COLORS)
    mat_oh   = one_hot(obj['material'], MATERIALS)
    shape_oh = one_hot(obj['shape'],    SHAPES)

    if len(traj) >= 2:
        vx = traj[-1][0] - traj[-2][0]
        vy = traj[-1][1] - traj[-2][1]
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


def exits_after(obj_id, frame, in_outs):
    """
    Corrected labeling function: True if object exits at ANY point
    after the collision frame. (The V1 lookahead=20 bug is removed.)
    """
    for event in in_outs:
        if (event.get('object') == obj_id
                and event.get('type') == 'out'
                and event.get('frame', 0) >= frame):
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

        # ── Exit label ────────────────────────────────────────
        label = 1.0 if any(
            exits_after(oid, frame, in_outs) for oid in obj_ids
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