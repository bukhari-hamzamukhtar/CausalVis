# src/data/build_dataset.py
import json
import os
import torch
from torch_geometric.data import Data
from pycocotools import mask as mask_utils
import numpy as np

COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
MATERIALS = ['rubber', 'metal']
SHAPES = ['cube', 'sphere', 'cylinder']

def one_hot(val, categories):
    vec = [0.0] * len(categories)
    if val in categories:
        vec[categories.index(val)] = 1.0
    return vec

def get_node_features(obj, traj):
    color_oh = one_hot(obj['color'], COLORS)
    mat_oh = one_hot(obj['material'], MATERIALS)
    shape_oh = one_hot(obj['shape'], SHAPES)
    
    if len(traj) >= 2:
        vx = traj[-1][0] - traj[-2][0]
        vy = traj[-1][1] - traj[-2][1]
        px, py = traj[-1]
    elif len(traj) == 1:
        vx, vy = 0.0, 0.0
        px, py = traj[0]
    else:
        vx, vy, px, py = 0.0, 0.0, 0.0, 0.0
    
    return color_oh + mat_oh + shape_oh + [px / 480, py / 320, vx, vy]

def exits_soon(obj_id, frame, in_outs):
    """
    Checks if the object exits the scene AT ANY POINT after the collision.
    We removed the strict lookahead window because objects can take 
    a long time to roll off the table.
    """
    for event in in_outs:
        if event['object'] == obj_id and event['type'] == 'out':
            # As long as the exit happens AFTER the collision frame, it's a 1.
            if event['frame'] >= frame:
                return True
    return False

def process_video(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    gt = data['ground_truth']
    objects = {o['id']: o for o in gt['objects']}
    collisions = gt['collisions']
    in_outs = gt.get('in_outs', [])
    
    positions = {}
    for frame in data['frames']:
        fi = frame['frame_index']
        positions[fi] = {}
        for det in frame['objects']:
            rle = det['mask']
            binary_mask = mask_utils.decode(rle)
            ys, xs = np.where(binary_mask)
            if len(xs) > 0:
                key = f"{det['color']}_{det['material']}_{det['shape']}"
                positions[fi][key] = (float(xs.mean()), float(ys.mean()))
    
    samples = []
    for event in collisions:
        frame = event['frame']
        obj_ids = event['object']
        if len(obj_ids) != 2:
            continue
        
        node_feats = []
        for oid in obj_ids:
            obj = objects[oid]
            key = f"{obj['color']}_{obj['material']}_{obj['shape']}"
            traj = [positions[f][key] for f in range(max(0, frame-5), frame+1)
                    if f in positions and key in positions[f]]
            node_feats.append(get_node_features(obj, traj))
        
        x = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        
        label = 1.0 if any(exits_soon(oid, frame, in_outs) for oid in obj_ids) else 0.0
        y = torch.tensor([label], dtype=torch.float)
        
        samples.append(Data(x=x, edge_index=edge_index, y=y))
    
    return samples

def build_full_dataset(data_dir, out_path):
    all_samples = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files.")
    
    for i, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        try:
            samples = process_video(path)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Skipped {fname}: {e}")
        if i % 50 == 0:
            print(f"Processed {i}/{len(files)} videos, {len(all_samples)} samples so far")
    
    torch.save(all_samples, out_path)
    print(f"Saved {len(all_samples)} graph samples to {out_path}")

if __name__ == "__main__":
    build_full_dataset(
        r"G:\CausalVis\data\processed_proposals",
        r"G:\CausalVis\data\causal_dataset.pt"
    )