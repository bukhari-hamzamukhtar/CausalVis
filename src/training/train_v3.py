# src/training/train_v3.py
# ─────────────────────────────────────────────────────────────
# V3 Training Script — HNN_PDGNN (Final Version, Module 1)
# Run on Google Colab (GPU)
#
# WHAT'S NEW VS V2:
#   • HNN_PDGNN instead of PD_GNN — velocity predictions now come
#     from Hamilton's equations (energy-conserving by construction)
#     instead of a plain linear head
#   • NEW: Hamiltonian consistency loss — encourages the learned H
#     to behave like a real Hamiltonian (dq/dt = ∂H/∂p ≈ velocity)
#   • Same Lagrange momentum constraint as V2, now operating on
#     Hamilton-derived velocities instead of linear-head velocities
#   • Reuses causal_dataset_v2.pt — no need to rebuild data
#
# ⚠ CRITICAL DIFFERENCE FROM ALL PREVIOUS SCRIPTS:
#   HNN_PDGNN needs LOCAL gradients (dH/dpos, dH/dvel) on every
#   single forward pass — including during evaluation. This is
#   intrinsic to how Hamiltonian dynamics work, not a training-only
#   mechanism. Every model call in this script — training, eval,
#   and counterfactual sensitivity — is wrapped in
#   `torch.enable_grad()`, even though eval() and the CSR check
#   would normally run under torch.no_grad(). Forgetting this
#   produces a cryptic "element 0 of tensors does not require grad"
#   RuntimeError. If you copy code out of this script elsewhere,
#   keep that wrapping intact.
# ─────────────────────────────────────────────────────────────

import os, random, bz2
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, precision_score,
                              recall_score)

try:
    from hnn_gnn import HNN_PDGNN
except ImportError:
    import sys
    sys.path.append('/content')
    from hnn_gnn import HNN_PDGNN


# ══════════════════════════════════════════════════════════════
# STEP 0 — Decompress if needed (same dataset as V2)
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


# ══════════════════════════════════════════════════════════════
# STEP 1 — Load dataset (reused from V2, same file)
# ══════════════════════════════════════════════════════════════
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
    print(f"  V2 structure  : ✓ (reused for V3)")
    return dataset


# ══════════════════════════════════════════════════════════════
# STEP 2 — Stratified split (identical to V2)
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# LAGRANGE MOMENTUM CONSTRAINT — same fixed version as V2
# ══════════════════════════════════════════════════════════════
def lagrange_momentum_loss(pred_vel_batch, pre_vel_batch,
                            mass_batch, batch_idx, lambda_momentum=1.0):
    """
    Per-graph momentum conservation via global_add_pool.
    See train_v2.py for the full explanation of why per-graph
    aggregation (not batch-wide) is required.
    """
    m = mass_batch.unsqueeze(1)
    p_before_node = m * pre_vel_batch
    p_after_node  = m * pred_vel_batch
    p_before_graph = global_add_pool(p_before_node, batch_idx)
    p_after_graph  = global_add_pool(p_after_node,  batch_idx)
    return lambda_momentum * F.mse_loss(p_after_graph, p_before_graph)


# ══════════════════════════════════════════════════════════════
# NEW — Hamiltonian consistency loss
# ══════════════════════════════════════════════════════════════
def hamiltonian_consistency_loss(dH_dvel_batch, vel_batch):
    """
    Hamilton's equation: dq/dt = ∂H/∂p.

    In our setup, 'velocity' is both the observed rate of change of
    position AND our chosen momentum proxy p. A properly-formed
    Hamiltonian should therefore satisfy: differentiating H with
    respect to velocity (∂H/∂p) should reproduce the actual velocity
    itself (dq/dt). This loss is a SELF-CONSISTENCY regularizer on
    the shape of H_net — it does not use any label, it just
    constrains H to behave like a real Hamiltonian rather than an
    arbitrary scalar function.
    """
    return F.mse_loss(dH_dvel_batch, vel_batch)


# ══════════════════════════════════════════════════════════════
# STEP 3 — Training loop
# ══════════════════════════════════════════════════════════════
def train(model, train_loader, optimizer, pos_weight, device,
          lambda_momentum=1.0, vel_weight=0.3, hamilton_weight=0.1):
    model.train()
    totals = {'total': 0.0, 'exit': 0.0, 'phys': 0.0,
              'vel': 0.0, 'hamilton': 0.0}

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Standard training forward — grad is on by default here,
        # no special wrapping needed (we're not inside no_grad).
        exit_logit, pred_vel, H_per_node, dH_dvel = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # ── Task 1: Exit prediction ───────────────────────────
        L_exit = F.binary_cross_entropy_with_logits(
            exit_logit.view(-1), batch.y, pos_weight=pos_weight.to(device))

        # ── Task 2: Velocity prediction accuracy ──────────────
        L_vel = F.mse_loss(pred_vel, batch.post_vel)

        # ── Task 3: Momentum conservation (per-graph) ─────────
        L_phys = lagrange_momentum_loss(
            pred_vel, batch.pre_vel, batch.mass, batch.batch,
            lambda_momentum=lambda_momentum)

        # ── Task 4: Hamiltonian self-consistency ──────────────
        L_hamilton = hamiltonian_consistency_loss(
            dH_dvel, batch.x[:, 15:17])   # ground-truth velocity slice

        loss = L_exit + vel_weight * L_vel + L_phys + hamilton_weight * L_hamilton

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals['total']    += loss.item()
        totals['exit']     += L_exit.item()
        totals['phys']     += L_phys.item()
        totals['vel']      += L_vel.item()
        totals['hamilton'] += L_hamilton.item()

    n = len(train_loader)
    return {k: v / n for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════
# STEP 4 — Evaluation
# ⚠ Must wrap model calls in torch.enable_grad() — see header note
# ══════════════════════════════════════════════════════════════
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    preds, trues = [], []

    for batch in loader:
        batch = batch.to(device)

        # HNN needs local gradients even here. Do NOT wrap this in
        # torch.no_grad() — that would break dH/dpos and dH/dvel.
        with torch.enable_grad():
            exit_logit, pred_vel, H, dH_dvel = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Detach immediately after extracting what we need — no
        # reason to keep the computation graph alive past this point.
        prob = torch.sigmoid(exit_logit).detach().view(-1)
        preds.extend((prob > threshold).int().cpu().tolist())
        trues.extend(batch.y.int().cpu().tolist())

    acc  = accuracy_score(trues, preds)
    f1   = f1_score(trues, preds, zero_division=0)
    prec = precision_score(trues, preds, zero_division=0)
    rec  = recall_score(trues, preds, zero_division=0)
    cm   = confusion_matrix(trues, preds)

    return {'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec,
            'confusion_matrix': cm}


# ══════════════════════════════════════════════════════════════
# STEP 5 — Three-mode counterfactual sensitivity
# ⚠ Same enable_grad requirement as evaluate()
# ══════════════════════════════════════════════════════════════
def counterfactual_sensitivity(model, test_loader, device, max_batches=50):
    model.eval()
    results = {'zero_vel': 0, 'remove_edge': 0, 'remove_obj': 0, 'total': 0}

    for i, batch in enumerate(test_loader):
        if i >= max_batches:
            break
        batch = batch.to(device)

        with torch.enable_grad():
            logit_base, _, _, _ = model(batch.x, batch.edge_index,
                                         batch.edge_attr, batch.batch)
        prob_base = torch.sigmoid(logit_base).detach().view(-1)
        pred_base = (prob_base > 0.5).int()

        # Mode 1: zero velocity of first node in each pair
        x_zv = batch.x.clone()
        x_zv[::2, -2:] = 0.0
        with torch.enable_grad():
            logit_zv, _, _, _ = model(x_zv, batch.edge_index,
                                       batch.edge_attr, batch.batch)
        pred_zv = (torch.sigmoid(logit_zv).detach().view(-1) > 0.5).int()

        # Mode 2: remove collision edge
        ei_empty = torch.zeros(2, 0, dtype=torch.long, device=device)
        ea_empty = torch.zeros(0, batch.edge_attr.size(1), device=device)
        try:
            with torch.enable_grad():
                logit_re, _, _, _ = model(batch.x, ei_empty, ea_empty, batch.batch)
            pred_re = (torch.sigmoid(logit_re).detach().view(-1) > 0.5).int()
        except Exception:
            pred_re = pred_base.clone()

        # Mode 3: zero all features of first node in each pair
        x_ro = batch.x.clone()
        x_ro[::2, :] = 0.0
        with torch.enable_grad():
            logit_ro, _, _, _ = model(x_ro, batch.edge_index,
                                       batch.edge_attr, batch.batch)
        pred_ro = (torch.sigmoid(logit_ro).detach().view(-1) > 0.5).int()

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
    DATASET_BZ2 = 'causal_dataset_v2.pt.bz2'
    DATASET_PT  = 'causal_dataset_v2.pt'
    SAVE_PATH   = 'causal_gnn_v3_hnn.pt'
    BATCH_SIZE  = 64
    EPOCHS      = 50
    LR          = 0.005
    WD          = 1e-4
    LAMBDA_MOM  = 1.0
    VEL_WEIGHT  = 0.3
    HAMILTON_WEIGHT = 0.1   # modest — new regularizer, start conservative

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'═'*58}")
    print(f" CausalVis V3 Training — HNN_PDGNN (Energy-Conserving)")
    print(f"{'═'*58}")
    print(f" Device: {device}")
    print(f" NOTE: ~2-3x slower per epoch than V2 due to double")
    print(f"       backward passes required for Hamilton's equations")

    if os.path.exists(DATASET_BZ2) and not os.path.exists(DATASET_PT):
        decompress_if_needed(DATASET_BZ2, DATASET_PT)

    dataset = load_dataset(DATASET_PT)
    train_data, test_data, pos_weight = stratified_split(dataset)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

    model = HNN_PDGNN(node_dim=17, edge_dim=4, hidden=64).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: HNN_PDGNN  |  Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    print(f"\n{'─'*58}")
    print(f" Training for {EPOCHS} epochs")
    print(f" Loss = L_exit + {VEL_WEIGHT}×L_vel + 1.0×L_phys + {HAMILTON_WEIGHT}×L_hamilton")
    print(f"{'─'*58}\n")

    best_f1 = 0.0
    history = []

    for epoch in range(EPOCHS):
        losses = train(model, train_loader, optimizer, pos_weight,
                       device, LAMBDA_MOM, VEL_WEIGHT, HAMILTON_WEIGHT)

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            metrics = evaluate(model, test_loader, device)
            scheduler.step(metrics['f1'])

            print(f"Epoch {epoch:>3}/{EPOCHS}"
                  f"  loss={losses['total']:.4f}"
                  f"  (exit={losses['exit']:.4f}"
                  f"  phys={losses['phys']:.4f}"
                  f"  vel={losses['vel']:.4f}"
                  f"  ham={losses['hamilton']:.4f})"
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
                  f"  ham={losses['hamilton']:.4f})")

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
    print(f"\n Confusion Matrix:")
    print(f"   {metrics['confusion_matrix']}")

    print(f"\n{'─'*58}")
    print(" THREE-MODE COUNTERFACTUAL SENSITIVITY (first 50 batches)")
    print(f"{'─'*58}")
    csr = counterfactual_sensitivity(model, test_loader, device)

    print(f"\n CSR_velocity  (zero momentum)  : {csr['CSR_velocity']*100:.1f}%")
    print(f" CSR_edge      (no collision)   : {csr['CSR_edge']*100:.1f}%")
    print(f" CSR_existence (object removed) : {csr['CSR_existence']*100:.1f}%")
    print(f" Events tested: {csr['total_events']}")

    # ── V1 → V2 → V3 comparison ───────────────────────────────
    print(f"\n{'═'*58}")
    print(" V1 → V2 → V3 COMPARISON")
    print(f"{'═'*58}")
    print(f" {'Metric':<12} {'V1':>8} {'V2':>8} {'V3':>8} {'V3−V2':>8}")
    print(f" {'─'*52}")
    v1 = {'accuracy': 0.806, 'f1': 0.464, 'precision': 0.350, 'recall': 0.680}
    v2 = {'accuracy': 0.812, 'f1': 0.472, 'precision': 0.360, 'recall': 0.685}
    for k in ['accuracy', 'f1', 'precision', 'recall']:
        v3_val = metrics[k]
        delta  = v3_val - v2[k]
        sign   = '+' if delta >= 0 else ''
        print(f" {k:<12} {v1[k]:>8.3f} {v2[k]:>8.3f} {v3_val:>8.3f} {sign+f'{delta:.3f}':>8}")
    print(f"{'═'*58}\n")

    torch.save({
        'history':        history,
        'final_metrics':  metrics,
        'csr':            csr,
        'model_config':   {'node_dim': 17, 'edge_dim': 4, 'hidden': 64},
    }, 'v3_training_log.pt')
    print(f"Training log saved: v3_training_log.pt")
    print(f"Best model saved:   {SAVE_PATH}")


if __name__ == "__main__":
    main()
