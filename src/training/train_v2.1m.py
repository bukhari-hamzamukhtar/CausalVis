# src/training/train_v2_masked.py
# ─────────────────────────────────────────────────────────────
# V2 + Object Masking Robustness Training (corrected version)
#
# CORRECTION VS THE ORIGINAL ROADMAP SKETCH:
#   The original idea was: mask a node's features, then train the
#   masked prediction to MATCH the full-information prediction.
#   That objective is backwards — it explicitly teaches the network
#   to be INSENSITIVE to object removal, which directly undermines
#   the Counterfactual Sensitivity Rate (CSR) validated in V2/V3.
#
#   CORRECTED APPROACH:
#     The masked forward pass is supervised against the TRUE
#     ground-truth exit label (batch.y), not against the unmasked
#     prediction. This is a standard feature-dropout / cutout-style
#     regularizer: it forces the network to build correct
#     predictions from partial evidence (improving robustness and
#     generalization) WITHOUT teaching it to ignore causally
#     relevant objects.
#
#   Physics losses (L_vel, L_phys) are computed on the CLEAN,
#   unmasked forward pass only — masking the input would corrupt
#   the physical meaning of pre_vel/post_vel/momentum targets,
#   which describe the TRUE trajectory, not an artificially
#   degraded one.
#
# WHAT THIS ADDS ON TOP OF train_v2.py:
#   • A second, masked forward pass per batch (mask one node's
#     velocity per graph, at random, with probability mask_prob)
#   • A new loss term: L_mask_exit — exit prediction supervised
#     under partial information
#   • Everything else (PD_GNN, Lagrange constraint, edge features)
#     is identical to train_v2.py
# ─────────────────────────────────────────────────────────────

import os, sys, random, bz2
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, precision_score,
                              recall_score)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.pd_gnn import PD_GNN


# ══════════════════════════════════════════════════════════════
# Reused from train_v2.py — decompress, load, split
# ══════════════════════════════════════════════════════════════
def decompress_if_needed(bz2_path, out_path):
    if os.path.exists(out_path):
        print(f"  {out_path} already exists. Skipping decompression.")
        return
    print(f"  Decompressing {bz2_path}...")
    with bz2.open(bz2_path, 'rb') as fi, open(out_path, 'wb') as fo:
        for chunk in iter(lambda: fi.read(100 * 1024), b''):
            fo.write(chunk)
    print("  Done.")


def load_dataset(pt_path):
    print(f"\nLoading dataset from {pt_path}...")
    dataset = torch.load(pt_path, weights_only=False)
    print(f"  Total samples : {len(dataset):,}")
    sample = dataset[0]
    assert hasattr(sample, 'edge_attr'), "Missing edge_attr"
    assert hasattr(sample, 'pre_vel'),   "Missing pre_vel"
    assert hasattr(sample, 'post_vel'),  "Missing post_vel"
    print(f"  Node features : {sample.x.shape[-1]}D")
    print(f"  Edge features : {sample.edge_attr.shape[-1]}D")
    return dataset


def stratified_split(dataset, seed=42):
    random.seed(seed)
    pos = [d for d in dataset if d.y.item() == 1]
    neg = [d for d in dataset if d.y.item() == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    pos_split = int(0.8 * len(pos))
    neg_split = int(0.8 * len(neg))

    train = pos[:pos_split] + neg[:neg_split]
    test  = pos[pos_split:] + neg[neg_split:]
    random.shuffle(train)

    print(f"\nDataset split (80/20 stratified):")
    print(f"  Train: {len(train):,}  ({pos_split} pos / {neg_split} neg)")
    print(f"  Test:  {len(test):,}   ({len(pos)-pos_split} pos / {len(neg)-neg_split} neg)")

    pos_weight = torch.tensor([neg_split / max(pos_split, 1)], dtype=torch.float)
    print(f"  pos_weight: {pos_weight.item():.2f}")
    return train, test, pos_weight


def lagrange_momentum_loss(pred_vel_batch, pre_vel_batch,
                            mass_batch, batch_idx, lambda_momentum=1.0):
    """Per-graph momentum conservation. See train_v2.py for full explanation."""
    m = mass_batch.unsqueeze(1)
    p_before_node = m * pre_vel_batch
    p_after_node  = m * pred_vel_batch
    p_before_graph = global_add_pool(p_before_node, batch_idx)
    p_after_graph  = global_add_pool(p_after_node,  batch_idx)
    return lambda_momentum * F.mse_loss(p_after_graph, p_before_graph)


# ══════════════════════════════════════════════════════════════
# NEW — Object masking augmentation
# ══════════════════════════════════════════════════════════════
def build_masked_input(x, num_graphs, device, mask_prob=0.3):
    """
    For each graph in the batch (exactly 2 nodes per graph, indices
    [2g, 2g+1] for graph g — guaranteed by how create_dataset.py
    builds each collision as a 2-node subgraph), randomly choose
    ONE of the two nodes to mask, with probability mask_prob per
    graph. Only the velocity slice is zeroed — NOT the full node —
    so the network still knows WHICH object it's reasoning about
    (color/material/shape identity is preserved), it just loses
    that object's kinematic information for this forward pass.

    Returns a NEW tensor; does not modify x in place.
    """
    x_masked = x.clone()

    which_node   = (torch.rand(num_graphs, device=device) < 0.5).long()  # 0 or 1
    graph_ids    = torch.arange(num_graphs, device=device)
    target_nodes = graph_ids * 2 + which_node

    apply_mask   = torch.rand(num_graphs, device=device) < mask_prob
    nodes_to_mask = target_nodes[apply_mask]

    if nodes_to_mask.numel() > 0:
        x_masked[nodes_to_mask, 15:17] = 0.0   # zero velocity only

    return x_masked


# ══════════════════════════════════════════════════════════════
# Training loop — now with masked robustness pass
# ══════════════════════════════════════════════════════════════
def train(model, train_loader, optimizer, pos_weight, device,
          lambda_momentum=1.0, vel_weight=0.3,
          mask_prob=0.3, mask_weight=0.5):
    model.train()
    totals = {'total': 0.0, 'exit': 0.0, 'phys': 0.0,
              'vel': 0.0, 'mask_exit': 0.0}

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # ── Clean forward pass: primary task + physics ────────
        # (physics losses MUST use clean, unmasked kinematics —
        # masking here would corrupt what pre_vel/post_vel mean)
        exit_logit, pred_vel = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        L_exit = F.binary_cross_entropy_with_logits(
            exit_logit.view(-1), batch.y, pos_weight=pos_weight.to(device))
        L_vel  = F.mse_loss(pred_vel, batch.post_vel)
        L_phys = lagrange_momentum_loss(
            pred_vel, batch.pre_vel, batch.mass, batch.batch,
            lambda_momentum=lambda_momentum)

        # ── Masked forward pass: robustness augmentation ──────
        # Supervised against the TRUE label — NOT against the
        # clean prediction. See module docstring for why.
        num_graphs = batch.y.size(0)
        x_masked = build_masked_input(batch.x, num_graphs, device, mask_prob)

        exit_logit_masked, _ = model(
            x_masked, batch.edge_index, batch.edge_attr, batch.batch)
        L_mask_exit = F.binary_cross_entropy_with_logits(
            exit_logit_masked.view(-1), batch.y, pos_weight=pos_weight.to(device))

        loss = L_exit + vel_weight * L_vel + L_phys + mask_weight * L_mask_exit

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals['total']     += loss.item()
        totals['exit']      += L_exit.item()
        totals['phys']      += L_phys.item()
        totals['vel']       += L_vel.item()
        totals['mask_exit'] += L_mask_exit.item()

    n = len(train_loader)
    return {k: v / n for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════
# Evaluation — identical to train_v2.py (clean input only; we
# only want to know real-world performance, not masked performance)
# ══════════════════════════════════════════════════════════════
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logit, _ = model(batch.x, batch.edge_index,
                              batch.edge_attr, batch.batch)
            prob = torch.sigmoid(logit).view(-1)
            preds.extend((prob > threshold).int().cpu().tolist())
            trues.extend(batch.y.int().cpu().tolist())

    return {
        'accuracy':  accuracy_score(trues, preds),
        'f1':        f1_score(trues, preds, zero_division=0),
        'precision': precision_score(trues, preds, zero_division=0),
        'recall':    recall_score(trues, preds, zero_division=0),
        'confusion_matrix': confusion_matrix(trues, preds),
    }


def counterfactual_sensitivity(model, test_loader, device, max_batches=50):
    """Same three-mode CSR check as train_v2.py — unchanged."""
    model.eval()
    results = {'zero_vel': 0, 'remove_edge': 0, 'remove_obj': 0, 'total': 0}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            batch = batch.to(device)

            logit_base, _ = model(batch.x, batch.edge_index,
                                   batch.edge_attr, batch.batch)
            pred_base = (torch.sigmoid(logit_base).view(-1) > 0.5).int()

            x_zv = batch.x.clone()
            x_zv[::2, -2:] = 0.0
            logit_zv, _ = model(x_zv, batch.edge_index,
                                 batch.edge_attr, batch.batch)
            pred_zv = (torch.sigmoid(logit_zv).view(-1) > 0.5).int()

            ei_empty = torch.zeros(2, 0, dtype=torch.long, device=device)
            ea_empty = torch.zeros(0, batch.edge_attr.size(1), device=device)
            try:
                logit_re, _ = model(batch.x, ei_empty, ea_empty, batch.batch)
                pred_re = (torch.sigmoid(logit_re).view(-1) > 0.5).int()
            except Exception:
                pred_re = pred_base.clone()

            x_ro = batch.x.clone()
            x_ro[::2, :] = 0.0
            logit_ro, _ = model(x_ro, batch.edge_index,
                                 batch.edge_attr, batch.batch)
            pred_ro = (torch.sigmoid(logit_ro).view(-1) > 0.5).int()

            b = pred_base.size(0)
            results['total']       += b
            results['zero_vel']    += (pred_zv != pred_base).sum().item()
            results['remove_edge'] += (pred_re != pred_base).sum().item()
            results['remove_obj']  += (pred_ro != pred_base).sum().item()

    t = max(results['total'], 1)
    return {
        'CSR_velocity':  results['zero_vel']    / t,
        'CSR_edge':      results['remove_edge'] / t,
        'CSR_existence': results['remove_obj']  / t,
        'total_events':  results['total'],
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    DATASET_BZ2 = os.path.join(ROOT_DIR, 'data', 'causal_dataset_v2.pt.bz2')
    DATASET_PT  = os.path.join(ROOT_DIR, 'data', 'causal_dataset_v2.pt')
    SAVE_PATH   = os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_v2_masked.pt')
    BATCH_SIZE  = 64
    EPOCHS      = 50
    LR          = 0.005
    WD          = 1e-4
    LAMBDA_MOM  = 1.0
    VEL_WEIGHT  = 0.3
    MASK_PROB   = 0.3    # fraction of graphs that get one node masked per batch
    MASK_WEIGHT = 0.5    # weight of the robustness loss term

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'═'*58}")
    print(f" CausalVis V2+Masking Training — Robustness Augmentation")
    print(f"{'═'*58}")
    print(f" Device: {device}")

    if os.path.exists(DATASET_BZ2) and not os.path.exists(DATASET_PT):
        decompress_if_needed(DATASET_BZ2, DATASET_PT)

    dataset = load_dataset(DATASET_PT)
    train_data, test_data, pos_weight = stratified_split(dataset)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

    model = PD_GNN(node_dim=17, edge_dim=4, hidden=64).to(device)
    print(f"\nModel: PD_GNN  |  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    print(f"\n{'─'*58}")
    print(f" Training for {EPOCHS} epochs")
    print(f" Loss = L_exit + {VEL_WEIGHT}×L_vel + 1.0×L_phys + {MASK_WEIGHT}×L_mask_exit")
    print(f" mask_prob={MASK_PROB} (fraction of graphs with one node masked)")
    print(f"{'─'*58}\n")

    best_f1 = 0.0
    history = []

    for epoch in range(EPOCHS):
        losses = train(model, train_loader, optimizer, pos_weight, device,
                       LAMBDA_MOM, VEL_WEIGHT, MASK_PROB, MASK_WEIGHT)

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            metrics = evaluate(model, test_loader, device)
            scheduler.step(metrics['f1'])

            print(f"Epoch {epoch:>3}/{EPOCHS}"
                  f"  loss={losses['total']:.4f}"
                  f"  (exit={losses['exit']:.4f}"
                  f"  phys={losses['phys']:.4f}"
                  f"  vel={losses['vel']:.4f}"
                  f"  mask={losses['mask_exit']:.4f})"
                  f"  │  acc={metrics['accuracy']:.3f}"
                  f"  f1={metrics['f1']:.3f}"
                  f"  prec={metrics['precision']:.3f}"
                  f"  rec={metrics['recall']:.3f}")

            history.append({'epoch': epoch, **losses, **metrics})

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"              ✓ New best F1 = {best_f1:.4f} — saved")
        else:
            print(f"Epoch {epoch:>3}/{EPOCHS}"
                  f"  loss={losses['total']:.4f}"
                  f"  (exit={losses['exit']:.4f}"
                  f"  phys={losses['phys']:.4f}"
                  f"  vel={losses['vel']:.4f}"
                  f"  mask={losses['mask_exit']:.4f})")

    # ── Final evaluation ──────────────────────────────────────
    print(f"\n{'═'*58}")
    print(" FINAL EVALUATION — Loading best checkpoint")
    print(f"{'═'*58}")
    model.load_state_dict(
        torch.load(SAVE_PATH, map_location=device, weights_only=True))
    metrics = evaluate(model, test_loader, device)

    print(f"\n Accuracy  : {metrics['accuracy']:.4f}")
    print(f" F1 Score  : {metrics['f1']:.4f}")
    print(f" Precision : {metrics['precision']:.4f}")
    print(f" Recall    : {metrics['recall']:.4f}")
    print(f"\n Confusion Matrix:\n   {metrics['confusion_matrix']}")

    print(f"\n{'─'*58}")
    print(" THREE-MODE COUNTERFACTUAL SENSITIVITY")
    print(f"{'─'*58}")
    csr = counterfactual_sensitivity(model, test_loader, device)
    print(f"\n CSR_velocity  : {csr['CSR_velocity']*100:.1f}%")
    print(f" CSR_edge      : {csr['CSR_edge']*100:.1f}%")
    print(f" CSR_existence : {csr['CSR_existence']*100:.1f}%")

    # ── V2 vs V2+Masking comparison ───────────────────────────
    print(f"\n{'═'*58}")
    print(" V2 → V2+MASKING COMPARISON")
    print(f"{'═'*58}")
    v2 = {'accuracy': 0.812, 'f1': 0.472, 'precision': 0.360, 'recall': 0.685}
    v2_csr = {'CSR_velocity': 0.230, 'CSR_edge': 0.212, 'CSR_existence': 0.216}
    print(f" {'Metric':<15} {'V2':>8} {'V2+Mask':>10} {'Δ':>8}")
    print(f" {'─'*45}")
    for k in ['accuracy', 'f1', 'precision', 'recall']:
        d = metrics[k] - v2[k]
        print(f" {k:<15} {v2[k]:>8.3f} {metrics[k]:>10.3f} {d:>+8.3f}")
    for k in ['CSR_velocity', 'CSR_edge', 'CSR_existence']:
        d = csr[k] - v2_csr[k]
        print(f" {k:<15} {v2_csr[k]*100:>7.1f}% {csr[k]*100:>9.1f}% {d*100:>+7.1f}%")
    print(f"{'═'*58}")
    print(" WATCH FOR: if CSR values DROPPED vs V2, the masking")
    print(" augmentation may still be teaching unwanted robustness")
    print(" at the cost of causal sensitivity — reduce MASK_WEIGHT")
    print(" and retry. If CSR held steady or rose while accuracy/F1")
    print(" improved, the augmentation is working as intended.")
    print(f"{'═'*58}\n")

    torch.save({'history': history, 'final_metrics': metrics, 'csr': csr},
               os.path.join(ROOT_DIR, 'src', 'models', 'v2_masked_training_log.pt'))
    print(f"Best model saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
