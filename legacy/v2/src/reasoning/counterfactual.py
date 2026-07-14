# src/reasoning/counterfactual.py
# ─────────────────────────────────────────────────────────────
# Three-mode counterfactual engine — V2 (PD_GNN) version.
#
# CORRECTNESS FIX VS THE V1 PORT:
#   V1's counterfactual engine only zeroed node feature slices in
#   x. That was correct for V1 because V1 had NO edge features —
#   the graph's only signal about "how fast are these objects
#   closing on each other" lived in the node velocities themselves.
#
#   V2 introduced edge_attr (distance, angle, closing speed, force
#   proxy) computed FROM the node positions/velocities at
#   construction time. If we now zero a node's velocity for a
#   counterfactual query but leave edge_attr untouched, the edge
#   still encodes the ORIGINAL (factual) closing speed — the
#   intervention leaks the pre-intervention physics right back in
#   through a side channel, silently corrupting the counterfactual.
#
#   Every intervention below calls recompute_edge_attr() after
#   modifying x, so the edge features are always consistent with
#   whatever the current (possibly counterfactual) node state is.
# ─────────────────────────────────────────────────────────────

import os, sys, copy
import torch

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.loader import (load_scene, build_event_subgraph,
                              compute_edge_features)
from src.models.pd_gnn import PD_GNN


def get_object_name(obj):
    return f"{obj['color']} {obj['material']} {obj['shape']}"


def recompute_edge_attr(x, edge_index, materials):
    """
    Rebuilds edge_attr from the CURRENT x tensor (post-intervention)
    using the exact same formula as create_dataset.py. See module
    docstring above for why this step cannot be skipped.
    """
    src, dst = edge_index
    if src.numel() == 0:
        return torch.zeros(0, 4)

    new_attrs = []
    for i in range(src.size(0)):
        s, d = src[i].item(), dst[i].item()
        pos_s = (x[s, 13].item(), x[s, 14].item())
        pos_d = (x[d, 13].item(), x[d, 14].item())
        vel_s = (x[s, 15].item(), x[s, 16].item())
        vel_d = (x[d, 15].item(), x[d, 16].item())
        new_attrs.append(
            compute_edge_features(pos_s, pos_d, vel_s, vel_d, materials[s]))

    return torch.tensor(new_attrs, dtype=torch.float)


def run_model(model, x, edge_index, edge_attr):
    """Single-graph inference. Returns exit probability (float)."""
    model.eval()
    with torch.no_grad():
        batch = torch.zeros(x.size(0), dtype=torch.long)
        exit_logit, _ = model(x, edge_index, edge_attr, batch)
        return torch.sigmoid(exit_logit).item()


def intervene(model, subgraph, objects, target_idx, mode='zero_velocity'):
    """
    Runs factual + counterfactual inference for one intervention.

    Parameters
    ----------
    subgraph   : dict from build_event_subgraph()
    objects    : {id: obj_dict} from load_scene()
    target_idx : 0 or 1 — which node in this subgraph to intervene on
    mode       : 'zero_velocity' | 'prevent_collision' | 'remove_object'
    """
    x          = subgraph['x']
    edge_index = subgraph['edge_index']
    edge_attr  = subgraph['edge_attr']
    obj_ids    = subgraph['obj_ids']
    materials  = subgraph['materials']

    affected_idx  = 1 - target_idx
    target_name   = get_object_name(objects[obj_ids[target_idx]])
    affected_name = get_object_name(objects[obj_ids[affected_idx]])

    # ── Factual ─────────────────────────────────────────────
    prob_factual = run_model(model, x, edge_index, edge_attr)

    # ── Counterfactual ──────────────────────────────────────
    cf_x          = x.clone()
    cf_edge_index = edge_index.clone()
    cf_edge_attr  = edge_attr.clone()

    if mode == 'zero_velocity':
        cf_x[target_idx, 15:17] = 0.0
        cf_edge_attr = recompute_edge_attr(cf_x, cf_edge_index, materials)
        description = f"[{target_name}] had zero velocity at impact"

    elif mode == 'prevent_collision':
        cf_edge_index = torch.zeros(2, 0, dtype=torch.long)
        cf_edge_attr  = torch.zeros(0, 4)
        description = f"[{target_name}] never collided with [{affected_name}]"

    elif mode == 'remove_object':
        cf_x[target_idx, :] = 0.0
        cf_edge_attr = recompute_edge_attr(cf_x, cf_edge_index, materials)
        description = f"[{target_name}] was not present in the scene"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    prob_cf = run_model(model, cf_x, cf_edge_index, cf_edge_attr)

    factual_outcome = "EXIT" if prob_factual > 0.5 else "STAY"
    cf_outcome      = "EXIT" if prob_cf      > 0.5 else "STAY"
    causal          = factual_outcome != cf_outcome

    if causal:
        rule = (f"IF {description}\n"
                f"THEN [{affected_name}] would {cf_outcome} instead of {factual_outcome}\n"
                f"BECAUSE [{target_name}]'s contribution WAS the determining factor\n"
                f"(Δprob = {abs(prob_cf - prob_factual):.3f})")
    else:
        rule = (f"IF {description}\n"
                f"THEN [{affected_name}] would still {cf_outcome} (unchanged)\n"
                f"BECAUSE [{target_name}] was NOT the sole determining factor\n"
                f"(Δprob = {abs(prob_cf - prob_factual):.3f})")

    return {
        'mode': mode, 'target': target_name, 'affected': affected_name,
        'factual_prob': prob_factual, 'cf_prob': prob_cf,
        'factual_outcome': factual_outcome, 'cf_outcome': cf_outcome,
        'is_causal': causal, 'rule': rule,
    }


def run_all_interventions(model, subgraph, objects, event):
    """Runs all three modes × both objects for one collision event."""
    obj_ids = subgraph['obj_ids']
    print(f"\n{'='*65}")
    print(f"COLLISION: Frame {event['frame']} | "
          f"{get_object_name(objects[obj_ids[0]])} ↔ "
          f"{get_object_name(objects[obj_ids[1]])}")
    print(f"{'='*65}")

    for mode in ['zero_velocity', 'prevent_collision', 'remove_object']:
        for idx in [0, 1]:
            r = intervene(model, subgraph, objects, idx, mode)
            status = "✅ CAUSAL" if r['is_causal'] else "❌ NOT SOLE CAUSE"
            print(f"\n[{mode.upper()} | Object {idx}] {status}")
            print(f"  Factual:        {r['affected']} → {r['factual_outcome']} "
                  f"(p={r['factual_prob']:.2f})")
            print(f"  Counterfactual: {r['affected']} → {r['cf_outcome']} "
                  f"(p={r['cf_prob']:.2f})")
            print(f"  Rule:\n    {r['rule'].replace(chr(10), chr(10)+'    ')}")


if __name__ == "__main__":
    model = PD_GNN(node_dim=17, edge_dim=4, hidden=64)
    model.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_v2.pt'),
        map_location='cpu'))

    json_path = os.path.join(
        ROOT_DIR, 'data', 'processed_proposals', 'sim_00000.json')
    objects, collisions, positions = load_scene(json_path)

    for event in collisions:
        subgraph = build_event_subgraph(objects, positions, event)
        run_all_interventions(model, subgraph, objects, event)
