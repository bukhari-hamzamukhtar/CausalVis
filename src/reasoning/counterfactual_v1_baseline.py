# src/reasoning/counterfactual.py

import os
import sys
import torch
import torch.nn.functional as F
import copy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pkg_paths import add_repo_root_to_path

ROOT_DIR = add_repo_root_to_path()

from src.data.loader import load_scene
from src.data.build_dataset import process_video

def get_object_name(obj):
    return f"{obj['color']} {obj['material']} {obj['shape']}"

def run_counterfactual(model, data, objects, event, question_obj_idx=0):
    """
    Asks: 'What if Object X had zero velocity at the moment of collision?'
    
    question_obj_idx: 0 or 1 — which of the two colliding objects to freeze.
    """
    obj_ids = event['object']
    
    # --- FACTUAL: what actually happens ---
    model.eval()
    with torch.no_grad():
        out_factual = torch.sigmoid(
            model.lin(
                F.relu(model.conv2(
                    F.relu(model.conv1(data.x, data.edge_index)),
                    data.edge_index
                )).mean(dim=0, keepdim=True)
            )
        ).item()
    
    # --- COUNTERFACTUAL: zero out velocity of chosen object ---
    cf_data = copy.deepcopy(data)
    cf_data.x[question_obj_idx, -2] = 0.0  # vx = 0
    cf_data.x[question_obj_idx, -1] = 0.0  # vy = 0
    
    with torch.no_grad():
        out_cf = torch.sigmoid(
            model.lin(
                F.relu(model.conv2(
                    F.relu(model.conv1(cf_data.x, cf_data.edge_index)),
                    cf_data.edge_index
                )).mean(dim=0, keepdim=True)
            )
        ).item()
    
    frozen_name = get_object_name(objects[obj_ids[question_obj_idx]])
    other_name  = get_object_name(objects[obj_ids[1 - question_obj_idx]])
    
    factual_outcome    = "EXIT" if out_factual > 0.5 else "STAY"
    cf_outcome         = "EXIT" if out_cf > 0.5 else "STAY"
    changed            = factual_outcome != cf_outcome
    
    print(f"\n--- COUNTERFACTUAL QUERY ---")
    print(f"What if [{frozen_name}] had zero velocity at collision?")
    print(f"  Factual outcome  → {other_name} would {factual_outcome} "
          f"(prob={out_factual:.2f})")
    print(f"  Counterfactual   → {other_name} would {cf_outcome}    "
          f"(prob={out_cf:.2f})")
    
    if changed:
        print(f"  ✅ CAUSAL: [{frozen_name}] WAS the cause. "
              f"Without its momentum, [{other_name}] would have {cf_outcome}.")
    else:
        print(f"  ❌ NOT SOLE CAUSE: Removing [{frozen_name}]'s velocity "
              f"did not change the outcome for [{other_name}].")
    
    return {
        'frozen_object': frozen_name,
        'affected_object': other_name,
        'factual': factual_outcome,
        'counterfactual': cf_outcome,
        'is_causal': changed
    }


if __name__ == "__main__":
    from src.models.gnn import CausalGNN
    
    model = CausalGNN()
    model.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_weighted.pt'),
        map_location='cpu'
    ))

    json_path = r"G:\CausalVis\data\processed_proposals\sim_00000.json"
    objects, collisions, _ = load_scene(json_path)
    samples = process_video(json_path)
    
    # Run counterfactual on every collision in the video
    for i, (event, data) in enumerate(zip(collisions, samples)):
        print(f"\n{'='*60}")
        print(f"COLLISION {i+1}: Frame {event['frame']}, "
              f"Objects {event['object']}")
        
        # Ask about both objects
        run_counterfactual(model, data, objects, event, question_obj_idx=0)
        run_counterfactual(model, data, objects, event, question_obj_idx=1)