import json, os
from pycocotools import mask as mask_utils
import numpy as np

def load_scene(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    gt = data['ground_truth']
    
    # Object attributes — your GNN node features
    objects = {obj['id']: obj for obj in gt['objects']}
    
    # Collision events — your subgraph boundaries
    collisions = gt['collisions']
    
    # Object positions per frame from masks
    positions = {}  # {frame_idx: {obj_color_key: (cx, cy)}}
    for frame in data['frames']:
        fi = frame['frame_index']
        positions[fi] = {}
        for det in frame['objects']:
            rle = det['mask']
            binary_mask = mask_utils.decode(rle)
            # Get centroid
            ys, xs = np.where(binary_mask)
            if len(xs) > 0:
                cx, cy = float(xs.mean()), float(ys.mean())
                key = f"{det['color']}_{det['material']}_{det['shape']}"
                positions[fi][key] = (cx, cy)
    
    return objects, collisions, positions

if __name__ == "__main__":
    # Pointing directly to your G drive setup
    json_path = r"G:\CausalVis\data\processed_proposals\sim_00000.json"
    
    if os.path.exists(json_path):
        print(f"Loading data from {json_path}...")
        objects, collisions, positions = load_scene(json_path)
        
        print("\n--- FOUND COLLISIONS ---")
        for col in collisions:
            print(col)
            
        print("\n--- POSITIONS AT FRAME 20 (Collision Frame) ---")
        # Let's peek at frame 20 since we know a collision happens there
        if 20 in positions:
            for obj_key, coords in positions[20].items():
                # Rounding the coords just to make it readable
                print(f"{obj_key}: ({round(coords[0], 2)}, {round(coords[1], 2)})")
    else:
        print(f"Error: Could not find {json_path}. Double-check your folder path!")