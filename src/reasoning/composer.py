import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pkg_paths import add_repo_root_to_path

ROOT_DIR = add_repo_root_to_path()

from src.data.loader import load_scene
from src.data.build_dataset import process_video

def get_object_name(obj):
    return f"{obj['color']} {obj['material']} {obj['shape']}"

def compose_video_narrative(json_path, model):
    """Takes a full video JSON, runs GNN on each collision subgraph,
    chains the rules into one human-readable causal narrative."""
    
    objects, collisions, positions = load_scene(json_path)
    samples = process_video(json_path)
    
    if not collisions:
        return "No collisions detected in this video."
    
    per_event_rules = []
    
    for i, (event, data) in enumerate(zip(collisions, samples)):
        frame = event['frame']
        obj_ids = event['object']
        obj0_name = get_object_name(objects[obj_ids[0]])
        obj1_name = get_object_name(objects[obj_ids[1]])
        
        model.eval()
        with torch.no_grad():
            x = F.relu(model.conv1(data.x, data.edge_index))
            x = F.relu(model.conv2(x, data.edge_index))
            prob = torch.sigmoid(model.lin(x.mean(dim=0, keepdim=True))).item()
        
        v0 = (data.x[0][-2]**2 + data.x[0][-1]**2).sqrt().item()
        v1 = (data.x[1][-2]**2 + data.x[1][-1]**2).sqrt().item()
        dominant_name = obj0_name if v0 > v1 else obj1_name
        affected_name = obj1_name if v0 > v1 else obj0_name
        
        outcome = "EXIT" if prob > 0.5 else "STAY"
        
        per_event_rules.append({
            'frame': frame,
            'dominant': dominant_name,
            'affected': affected_name,
            'outcome': outcome,
            'prob': prob,
            'obj_ids': obj_ids
        })
    
    # Chain the rules into a narrative
    print("\n" + "="*60)
    print("CAUSAL CHAIN NARRATIVE")
    print("="*60)
    
    for i, rule in enumerate(per_event_rules):
        prefix = f"[Event {i+1} | Frame {rule['frame']}]"
        
        if rule['outcome'] == "EXIT":
            line = (f"{prefix} {rule['dominant']} collided with "
                    f"{rule['affected']} → predicted to cause EXIT "
                    f"(confidence: {rule['prob']:.2f})")
        else:
            line = (f"{prefix} {rule['dominant']} collided with "
                    f"{rule['affected']} → insufficient force, "
                    f"predicted to STAY (confidence: {1-rule['prob']:.2f})")
        
        print(line)
        
        # Check if affected object appears in next event
        if i < len(per_event_rules) - 1:
            next_ids = per_event_rules[i+1]['obj_ids']
            if rule['obj_ids'][1] in next_ids or rule['obj_ids'][0] in next_ids:
                print(f"  ↓ {rule['affected']} carries momentum into next collision...")
    
    print("="*60)
    return per_event_rules


if __name__ == "__main__":
    from src.models.gnn import CausalGNN

    model = CausalGNN()
    model.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_weighted.pt'),
        map_location='cpu'
    ))

    compose_video_narrative(
        r"G:\CausalVis\data\processed_proposals\sim_00000.json",
        model
    )