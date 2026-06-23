"""
Counterfactual Intervention Engine (V2)
Three intervention modes corresponding to Module III token types:

  zero_velocity     — vx=0, vy=0 for target object
  prevent_collision — remove edge between the two objects
  remove_object     — zero ALL features of target object

Each mode asks a different counterfactual question.
"""

import os, sys, copy, torch, torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from src.data.loader import load_scene
from src.data.build_dataset import process_video

def get_object_name(obj):
    return f"{obj['color']} {obj['material']} {obj['shape']}"

def _forward(model, data):
    """Clean single-sample forward pass, returns probability."""
    model.eval()
    with torch.no_grad():
        x = F.relu(model.conv1(data.x, data.edge_index))
        x = F.relu(model.conv2(x, data.edge_index))
        return torch.sigmoid(model.lin(x.mean(dim=0, keepdim=True))).item()

def intervene(model, data, objects, event, target_idx, mode='zero_velocity'):
    """
    Run factual + counterfactual inference.

    Parameters
    ----------
    model       : trained CausalGNN
    data        : PyG Data object for this collision
    objects     : dict {id: obj_dict} from load_scene
    event       : collision event dict {'frame', 'object': [id0, id1]}
    target_idx  : 0 or 1 — which object to intervene on
    mode        : 'zero_velocity' | 'prevent_collision' | 'remove_object'
    """
    obj_ids      = event['object']
    target_name  = get_object_name(objects[obj_ids[target_idx]])
    affected_idx = 1 - target_idx
    affected_name = get_object_name(objects[obj_ids[affected_idx]])

    # --- Factual ---
    prob_factual = _forward(model, data)

    # --- Counterfactual ---
    cf = copy.deepcopy(data)

    if mode == 'zero_velocity':
        # "What if target had no momentum?"
        cf.x[target_idx, -2] = 0.0
        cf.x[target_idx, -1] = 0.0
        cf_description = f"[{target_name}] had zero velocity at impact"

    elif mode == 'prevent_collision':
        # "What if the collision never happened?" — remove the edge
        # edge_index is [[0,1],[1,0]], remove both directed edges
        cf.edge_index = torch.zeros((2, 0), dtype=torch.long)
        cf_description = f"[{target_name}] never collided with [{affected_name}]"

    elif mode == 'remove_object':
        # "What if target didn't exist?" — zero all features
        cf.x[target_idx, :] = 0.0
        cf_description = f"[{target_name}] was not present in the scene"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    prob_cf = _forward(model, cf)

    factual_outcome = "EXIT" if prob_factual > 0.5 else "STAY"
    cf_outcome      = "EXIT" if prob_cf      > 0.5 else "STAY"
    causal          = factual_outcome != cf_outcome

    # Build auditable causal rule
    if causal:
        rule = (f"IF {cf_description}\n"
                f"THEN [{affected_name}] would {cf_outcome} instead of {factual_outcome}\n"
                f"BECAUSE [{target_name}]'s contribution WAS the determining factor\n"
                f"(Δprob = {abs(prob_cf - prob_factual):.3f})")
    else:
        rule = (f"IF {cf_description}\n"
                f"THEN [{affected_name}] would still {cf_outcome} (unchanged)\n"
                f"BECAUSE [{target_name}] was NOT the sole determining factor\n"
                f"(Δprob = {abs(prob_cf - prob_factual):.3f})")

    return {
        'mode': mode,
        'target': target_name,
        'affected': affected_name,
        'factual_prob': prob_factual,
        'cf_prob': prob_cf,
        'factual_outcome': factual_outcome,
        'cf_outcome': cf_outcome,
        'is_causal': causal,
        'rule': rule
    }

def run_all_interventions(model, data, objects, event):
    """Run all three intervention types for one collision event."""
    obj_ids = event['object']
    print(f"\n{'='*65}")
    print(f"COLLISION: Frame {event['frame']} | "
          f"{get_object_name(objects[obj_ids[0]])} ↔ "
          f"{get_object_name(objects[obj_ids[1]])}")
    print(f"{'='*65}")

    for mode in ['zero_velocity', 'prevent_collision', 'remove_object']:
        for idx in [0, 1]:
            r = intervene(model, data, objects, event, idx, mode)
            status = "✅ CAUSAL" if r['is_causal'] else "❌ NOT SOLE CAUSE"
            print(f"\n[{mode.upper()} | Object {idx}] {status}")
            print(f"  Factual:        {r['affected']} → {r['factual_outcome']} "
                  f"(p={r['factual_prob']:.2f})")
            print(f"  Counterfactual: {r['affected']} → {r['cf_outcome']} "
                  f"(p={r['cf_prob']:.2f})")
            print(f"  Rule:\n    {r['rule'].replace(chr(10), chr(10)+'    ')}")


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

    for event, data in zip(collisions, samples):
        run_all_interventions(model, data, objects, event)