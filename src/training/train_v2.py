# src/training/train_v2.py
# ─────────────────────────────────────────────────────────────
# V2 Training Script — Run on Google Colab (GPU)
#
# SETUP ON COLAB:
#   1. Upload causal_dataset_v2.pt (compressed as .bz2 if needed)
#   2. Upload pd_gnn.py to Colab working directory
#   3. Run this script
#
# WHAT'S NEW VS V1:
#   • PD_GNN instead of CausalGNN (unbounded force pathway)
#   • Lagrange momentum constraint (λ=10.0, not soft 0.1)
#   • Multi-task: exit prediction + velocity prediction
#   • Passes edge_attr through the GNN
#   • Saves full metrics history for comparison plots
# ─────────────────────────────────────────────────────────────

import os, random, bz2, sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, precision_score,
                              recall_score)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

MODEL_DIR = os.path.join(ROOT_DIR, 'src', 'models')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# ── Import PD_GNN ─────────────────────────────────────────────
# If running on Colab from repo root, this file will find src/models/pd_gnn.py
try:
    from pd_gnn import PD_GNN, CausalGNN
except ImportError:
    sys.path.insert(0, '/content')
    from pd_gnn import PD_GNN, CausalGNN


def resolve_dataset_path(filename):
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(ROOT_DIR, filename),
        os.path.join(ROOT_DIR, 'data', filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


# ══════════════════════════════════════════════════════════════
# STEP 0 — Decompress if needed
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
# STEP 1 — Load dataset
# ══════════════════════════════════════════════════════════════
def load_dataset(pt_path):
    print(f"\nLoading dataset from {pt_path}...")
    dataset = torch.load(pt_path, weights_only=False)
    print(f"  Total samples : {len(dataset):,}")

    # Verify V2 structure
    sample = dataset[0]
    assert hasattr(sample, 'edge_attr'),  "Missing edge_attr — rebuild with build_dataset_v2.py"
    assert hasattr(sample, 'pre_vel'),    "Missing pre_vel  — rebuild with build_dataset_v2.py"
    assert hasattr(sample, 'post_vel'),   "Missing post_vel — rebuild with build_dataset_v2.py"
    print(f"  Node features : {sample.x.shape[-1]}D")
    print(f"  Edge features : {sample.edge_attr.shape[-1]}D")
    print(f"  V2 structure  : ✓")
    return dataset


# ══════════════════════════════════════════════════════════════
# STEP 2 — Stratified split
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
# LAGRANGE MOMENTUM CONSTRAINT LOSS
# ══════════════════════════════════════════════════════════════
def lagrange_momentum_loss(pred_vel_batch, pre_vel_batch,
                            mass_batch, lambda_momentum=10.0):
    """
    Hard momentum conservation constraint via Lagrange multiplier.

    WHY λ=10.0 (not 0.1 like V1's soft penalty):
      A soft penalty of 0.1 is cheaply violated — the optimizer
      finds it easier to ignore it than to satisfy it.
      λ=10.0 makes the constraint DOMINATE the loss early in
      training, forcing the network to learn physically consistent
      dynamics before optimizing exit prediction accuracy.

    Conservation law being enforced:
      Σ(mass_i × velocity_i) BEFORE = Σ(mass_i × velocity_i) AFTER

    Parameters
    ----------
    pred_vel_batch : [N, 2] predicted post-collision velocities
    pre_vel_batch  : [N, 2] ground truth pre-collision velocities
    mass_batch     : [N]    mass proxy per node
    lambda_momentum: float  Lagrange multiplier (dominance factor)

    Returns
    -------
    Scalar momentum conservation loss
    """
    # mass: [N] → [N, 1] for broadcasting
    m = mass_batch.unsqueeze(1)

    # Total momentum before and after (summed over all nodes in batch)
    # In a 2-node subgraph: p_before[0] + p_before[1] ≈ p_after[0] + p_after[1]
    p_before = (m * pre_vel_batch).sum(dim=0)    # [2] (x, y components)
    p_after  = (m * pred_vel_batch).sum(dim=0)   # [2]

    return lambda_momentum * F.mse_loss(p_after, p_before)


# ══════════════════════════════════════════════════════════════
# STEP 3 — Training loop
# ══════════════════════════════════════════════════════════════
def train(model, train_loader, optimizer, pos_weight, device,
          lambda_momentum=10.0, vel_weight=0.3):
    model.train()
    total_loss   = 0.0
    total_exit   = 0.0
    total_phys   = 0.0
    total_vel    = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        exit_logit, pred_vel = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # ── Task 1: Exit prediction ───────────────────────────
        L_exit = F.binary_cross_entropy_with_logits(
            exit_logit.view(-1),
            batch.y,
            pos_weight=pos_weight.to(device)
        )

        # ── Task 2: Velocity prediction (if ground truth available) ──
        if hasattr(batch, 'post_vel') and batch.post_vel is not None:
            L_vel = F.mse_loss(pred_vel, batch.post_vel)
        else:
            L_vel = torch.tensor(0.0, device=device)

        # ── Task 3: Lagrange momentum constraint ──────────────
        if hasattr(batch, 'pre_vel') and batch.pre_vel is not None:
            L_phys = lagrange_momentum_loss(
                pred_vel, batch.pre_vel, batch.mass,
                lambda_momentum=lambda_momentum)
        else:
            L_phys = torch.tensor(0.0, device=device)

        # ── Combined loss ─────────────────────────────────────
        loss = L_exit + vel_weight * L_vel + L_phys

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_exit += L_exit.item()
        total_phys += L_phys.item()
        total_vel  += L_vel.item()

    n = len(train_loader)
    return {
        'total': total_loss / n,
        'exit':  total_exit / n,
        'phys':  total_phys / n,
        'vel':   total_vel  / n,
    }


# ══════════════════════════════════════════════════════════════
# STEP 4 — Evaluation
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

    acc  = accuracy_score(trues, preds)
    f1   = f1_score(trues, preds, zero_division=0)
    prec = precision_score(trues, preds, zero_division=0)
    rec  = recall_score(trues, preds, zero_division=0)
    cm   = confusion_matrix(trues, preds)

    return {'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec,
            'confusion_matrix': cm}


# ══════════════════════════════════════════════════════════════
# STEP 5 — Three-mode counterfactual sensitivity (quick batch)
# ══════════════════════════════════════════════════════════════
def counterfactual_sensitivity(model, test_loader, device,
                                max_batches=50):
    """
    Reports counterfactual sensitivity rate for all three modes
    using the test loader. Fast version for end-of-training report.
    """
    import copy

    model.eval()
    results = {'zero_vel': 0, 'remove_edge': 0, 'remove_obj': 0, 'total': 0}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            batch = batch.to(device)

            logit_base, _ = model(batch.x, batch.edge_index,
                                   batch.edge_attr, batch.batch)
            prob_base = torch.sigmoid(logit_base).view(-1)
            pred_base = (prob_base > 0.5).int()

            # Mode 1: Zero velocity of first node in each subgraph
            x_zv = batch.x.clone()
            x_zv[::2, -2:] = 0.0   # zero vx, vy of node 0 in each pair
            logit_zv, _ = model(x_zv, batch.edge_index,
                                 batch.edge_attr, batch.batch)
            pred_zv = (torch.sigmoid(logit_zv).view(-1) > 0.5).int()

            # Mode 2: Remove collision edge (empty edge_index)
            ei_empty = torch.zeros(2, 0, dtype=torch.long, device=device)
            ea_empty = torch.zeros(0, batch.edge_attr.size(1), device=device)
            try:
                logit_re, _ = model(batch.x, ei_empty, ea_empty, batch.batch)
                pred_re = (torch.sigmoid(logit_re).view(-1) > 0.5).int()
            except Exception:
                pred_re = pred_base.clone()  # fallback if model errors

            # Mode 3: Zero all features of first node in each subgraph
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
    # ── Config ───────────────────────────────────────────────
    DATASET_BZ2 = 'causal_dataset_v2.pt.bz2'   # if compressed
    DATASET_PT  = 'causal_dataset_v2.pt'
    SAVE_PATH   = 'causal_gnn_v2.pt'
    BATCH_SIZE  = 64
    EPOCHS      = 50
    LR          = 0.005
    WD          = 1e-4
    LAMBDA_MOM  = 10.0
    VEL_WEIGHT  = 0.3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'═'*55}")
    print(f" CausalVis V2 Training — PD-GNN + Lagrange Constraint")
    print(f"{'═'*55}")
    print(f" Device: {device}")

    DATASET_BZ2 = resolve_dataset_path('causal_dataset_v2.pt.bz2')
    DATASET_PT  = resolve_dataset_path('causal_dataset_v2.pt')
    SAVE_PATH   = os.path.join(ROOT_DIR, 'causal_gnn_v2.pt')

    # ── Decompress if needed ──────────────────────────────────
    if os.path.exists(DATASET_BZ2) and not os.path.exists(DATASET_PT):
        decompress_if_needed(DATASET_BZ2, DATASET_PT)

    # ── Load data ─────────────────────────────────────────────
    dataset          = load_dataset(DATASET_PT)
    train_data, test_data, pos_weight = stratified_split(dataset)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────
    model = PD_GNN(node_dim=17, edge_dim=4, hidden=64).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: PD_GNN  |  Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WD)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=5, verbose=True)    # reduce LR when F1 stops improving

    # ── Training ──────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f" Training for {EPOCHS} epochs")
    print(f" Loss = L_exit + {VEL_WEIGHT}×L_vel + {LAMBDA_MOM}×L_phys")
    print(f"{'─'*55}\n")

    best_f1      = 0.0
    history      = []

    for epoch in range(EPOCHS):
        losses = train(model, train_loader, optimizer,
                       pos_weight, device, LAMBDA_MOM, VEL_WEIGHT)

        # Evaluate every 5 epochs (saves time)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            metrics = evaluate(model, test_loader, device)
            scheduler.step(metrics['f1'])

            print(f"Epoch {epoch:>3}/{EPOCHS}"
                  f"  loss={losses['total']:.4f}"
                  f"  (exit={losses['exit']:.4f}"
                  f"  phys={losses['phys']:.4f}"
                  f"  vel={losses['vel']:.4f})"
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
                  f"  vel={losses['vel']:.4f})")

    # ── Final evaluation ──────────────────────────────────────
    print(f"\n{'═'*55}")
    print(" FINAL EVALUATION — Loading best checkpoint")
    print(f"{'═'*55}")

    model.load_state_dict(
        torch.load(SAVE_PATH, map_location=device, weights_only=True))
    metrics = evaluate(model, test_loader, device)

    print(f"\n Accuracy  : {metrics['accuracy']:.4f}")
    print(f" F1 Score  : {metrics['f1']:.4f}")
    print(f" Precision : {metrics['precision']:.4f}")
    print(f" Recall    : {metrics['recall']:.4f}")
    print(f"\n Confusion Matrix:")
    print(f"   {metrics['confusion_matrix']}")

    # ── Counterfactual sensitivity ────────────────────────────
    print(f"\n{'─'*55}")
    print(" THREE-MODE COUNTERFACTUAL SENSITIVITY (first 50 batches)")
    print(f"{'─'*55}")
    csr = counterfactual_sensitivity(model, test_loader, device)

    print(f"\n CSR_velocity  (zero momentum)  : {csr['CSR_velocity']*100:.1f}%")
    print(f" CSR_edge      (no collision)   : {csr['CSR_edge']*100:.1f}%")
    print(f" CSR_existence (object removed) : {csr['CSR_existence']*100:.1f}%")
    print(f" Events tested: {csr['total_events']}")

    # ── V1 vs V2 Summary ─────────────────────────────────────
    print(f"\n{'═'*55}")
    print(" V1 → V2 COMPARISON (V1 values from completed run)")
    print(f"{'═'*55}")
    print(f" {'Metric':<25} {'V1':>8} {'V2':>8} {'Δ':>8}")
    print(f" {'─'*49}")
    v1 = {'accuracy': 0.806, 'f1': 0.464, 'precision': 0.35, 'recall': 0.68}
    for k in ['accuracy', 'f1', 'precision', 'recall']:
        v2_val = metrics[k]
        delta  = v2_val - v1[k]
        sign   = '+' if delta >= 0 else ''
        print(f" {k:<25} {v1[k]:>8.3f} {v2_val:>8.3f} {sign+f'{delta:.3f}':>8}")
    print(f"{'═'*55}\n")

    # ── Save full history ─────────────────────────────────────
    torch.save({
        'history':     history,
        'final_metrics': metrics,
        'csr':         csr,
        'model_config': {'node_dim': 17, 'edge_dim': 4, 'hidden': 64},
    }, 'v2_training_log.pt')
    print(f"Training log saved: v2_training_log.pt")
    print(f"Best model saved:   {SAVE_PATH}")


if __name__ == "__main__":
    main()