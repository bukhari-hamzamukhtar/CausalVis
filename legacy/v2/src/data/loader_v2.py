# src/data/loader.py
# ─────────────────────────────────────────────────────────────
# JSON parsing, RLE mask decoding, and a shared helper for
# building a single collision subgraph at INFERENCE time
# (matching the exact feature layout used by create_dataset.py
# so the reasoning layer sees the same distribution the V2 model
# was trained on).
#
# UPDATED: estimate_velocity() and build_node_features() now use
# a least-squares fit over the whole available window instead of
# a single last-two-frame difference, kept in sync with
# create_dataset.py's matching update (same reasoning: identical
# at exactly 2 points, more noise-robust with more available).
# ─────────────────────────────────────────────────────────────

import json
import numpy as np
import torch
from pycocotools import mask as mask_utils

# ── Vocabularies — MUST match create_dataset.py exactly ────────
COLORS    = ['gray','red','blue','green','brown','purple','cyan','yellow']
MATERIALS = ['rubber','metal']
SHAPES    = ['cube','sphere','cylinder']

MASS_PROXY  = {'rubber': 1.0, 'metal': 1.5}
RESTITUTION = {'rubber': 0.5, 'metal': 0.8}


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


def load_scene(json_path):
    """
    Parses a sim_XXXXX.json file.

    Returns
    -------
    objects    : {id: {'color', 'material', 'shape', ...}}
    collisions : [{'type', 'object': [id0, id1], 'frame'}, ...]
    positions  : {frame_idx: {color_mat_shape_key: (cx, cy)}}
    """
    with open(json_path) as f:
        data = json.load(f)

    gt = data['ground_truth']
    objects    = {o['id']: o for o in gt['objects']}
    collisions = gt['collisions']

    positions = {}
    for frame in data['frames']:
        fi = frame['frame_index']
        positions[fi] = {}
        for det in frame['objects']:
            key = f"{det['color']}_{det['material']}_{det['shape']}"
            c = get_centroid(det['mask'])
            if c:
                positions[fi][key] = c

    return objects, collisions, positions


def estimate_velocity(obj_key, frame, positions, window=5):
    """
    Least-squares fit over the whole window instead of a last-two-
    frame difference -- kept in sync with create_dataset.py's version
    (same reasoning: identical at 2 points, noise-robust with more).
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
    return float(np.polyfit(t_arr, xs, 1)[0]), float(np.polyfit(t_arr, ys, 1)[0])


def build_node_features(obj, traj):
    """17D node feature vector - identical layout to create_dataset.py."""
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

    return color_oh + mat_oh + shape_oh + [px / 480.0, py / 320.0, vx, vy]


def compute_edge_features(pos0, pos1, vel0, vel1, mat0):
    """
    4D edge attribute vector. Identical formula to create_dataset.py —
    kept in sync here so live inference matches training exactly.
    """
    dx = pos1[0] - pos0[0]
    dy = pos1[1] - pos0[1]
    dist = max(np.sqrt(dx**2 + dy**2), 1e-6)

    angle      = np.arctan2(dy, dx)
    angle_norm = (angle + np.pi) / (2 * np.pi)

    rel_vx = vel1[0] - vel0[0]
    rel_vy = vel1[1] - vel0[1]
    closing = -(rel_vx * dx + rel_vy * dy) / dist

    mass  = MASS_PROXY[mat0]
    force = mass * max(closing, 0.0)

    return [dist / 480.0, angle_norm, closing / 50.0, force / 5.0]


def build_event_subgraph(objects, positions, event, window=5):
    """
    Builds the (x, edge_index, edge_attr) tuple for ONE collision
    event, for live inference — same construction logic as
    create_dataset.py's process_video_v2(), but for a single event
    rather than a whole-dataset pass.

    Returns a dict:
        x          : [2, 17] tensor
        edge_index : [2, 2]  tensor  (bidirectional)
        edge_attr  : [2, 4]  tensor
        obj_ids    : [id0, id1]      original CLEVRER object ids
        materials  : [mat0, mat1]    for downstream mass-proxy lookups
    """
    frame   = event['frame']
    obj_ids = event['object']

    node_feats, pos_at_frame, vels, mats = [], [], [], []

    for oid in obj_ids:
        obj = objects[oid]
        key = f"{obj['color']}_{obj['material']}_{obj['shape']}"
        traj = [positions[f][key]
                for f in range(max(0, frame - window), frame + 1)
                if f in positions and key in positions[f]]

        node_feats.append(build_node_features(obj, traj))
        vels.append(estimate_velocity(key, frame, positions, window))
        mats.append(obj['material'])

        if frame in positions and key in positions[frame]:
            pos_at_frame.append(positions[frame][key])
        elif traj:
            pos_at_frame.append(traj[-1])
        else:
            pos_at_frame.append((0.0, 0.0))

    ef_0to1 = compute_edge_features(pos_at_frame[0], pos_at_frame[1],
                                     vels[0], vels[1], mats[0])
    ef_1to0 = compute_edge_features(pos_at_frame[1], pos_at_frame[0],
                                     vels[1], vels[0], mats[1])

    x          = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr  = torch.tensor([ef_0to1, ef_1to0], dtype=torch.float)

    return {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
        'obj_ids': obj_ids, 'materials': mats,
    }
