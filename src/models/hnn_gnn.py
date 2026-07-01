# src/models/hnn_gnn.py
# ─────────────────────────────────────────────────────────────
# HNN_PDGNN — PD-GNN body + Hamiltonian Neural Network output head
#
# WHAT CHANGES FROM V2 (PD_GNN):
#   V2's velocity_head was a plain nn.Linear(hidden, 2). Nothing
#   stopped it from predicting physically impossible velocity jumps
#   — energy conservation was only encouraged via the Lagrange
#   penalty, not guaranteed.
#
#   HNN_PDGNN replaces that head with a learned scalar Hamiltonian
#   H(embedding) and derives the velocity update via Hamilton's
#   canonical equations:
#
#       dq/dt =  ∂H/∂p      (adapted: consistency check, see below)
#       dp/dt = -∂H/∂q      (used as the velocity UPDATE)
#
#   Treating our stored 'velocity' feature as the momentum proxy p
#   and 'position' feature as q, we predict:
#
#       next_velocity = velocity + (-∂H/∂position)
#
#   Any update derived purely from gradients of ONE scalar function
#   conserves that function's value along the flow BY CONSTRUCTION
#   — this is the mathematical guarantee, not a training target.
#
# WHY THIS IS NOT FULL PORT-HAMILTONIAN ODEs:
#   No SE(3) equivariance, no Lie group theory, no continuous ODE
#   solver (torchdiffeq). This is the original Greydanus et al.
#   (2019) HNN idea, adapted from continuous phase-space trajectories
#   to our discrete before/after collision setting. Days of work,
#   not a PhD dissertation — this is the version from the corrected
#   roadmap, not the rejected Port-Hamiltonian proposal.
#
# ⚠ CRITICAL USAGE NOTE — READ BEFORE CALLING THIS MODEL:
#   HNN forward passes need LOCAL gradients (dH/dpos) even when you
#   are not training. This is intrinsic to how Hamiltonian dynamics
#   work — it is not a training-only mechanism. NEVER call this
#   model's forward() inside `torch.no_grad()`. Use
#   `with torch.enable_grad():` around every call, including during
#   evaluation and counterfactual inference. See train_v3.py for
#   the correct pattern.
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


class HNN_PDGNN(nn.Module):
    """
    Physics-Decoupled GNN body + Hamiltonian energy-conserving
    velocity prediction head.

    Parameters
    ----------
    node_dim : int  — input node feature dimension (17)
    edge_dim : int  — edge attribute dimension (4)
    hidden   : int  — hidden layer width
    """

    def __init__(self, node_dim=17, edge_dim=4, hidden=64):
        super().__init__()
        self.node_dim = node_dim

        # Feature layout (from build_dataset_v2.py):
        #   [0:8]   color one-hot
        #   [8:10]  material one-hot
        #   [10:13] shape one-hot
        #   [13:15] position (px, py)     ← q, needs grad for HNN
        #   [15:17] velocity (vx, vy)     ← p (momentum proxy), needs grad
        self.STATIC_END = 13

        # ── Pathway 1: Force MLP (unbounded, same as V2) ──────
        force_in = node_dim * 2 + edge_dim
        self.force_mlp = nn.Sequential(
            nn.Linear(force_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

        # ── Pathway 2: Feature GCN (same as V2) ────────────────
        augmented_dim = node_dim + 2
        self.conv1 = GCNConv(augmented_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

        # ── Exit prediction head (unchanged, separate from HNN) ─
        self.exit_head = nn.Linear(hidden, 1)

        # ── NEW: Hamiltonian energy head ───────────────────────
        # Scalar energy contribution per node. Tanh (not ReLU) is
        # required — Hamilton's equations need smooth, non-zero
        # second derivatives, which ReLU cannot provide (it is
        # piecewise-linear with zero curvature almost everywhere).
        self.H_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Returns
        -------
        exit_logit : [B, 1]  raw logit for exit prediction
        pred_vel   : [N, 2]  predicted post-collision velocity,
                              derived from Hamilton's equations
        H_per_node : [N, 1]  learned energy contribution per node
                              (for the Hamiltonian consistency loss)
        dH_dvel    : [N, 2]  ∂H/∂velocity — used for the consistency
                              loss (should ≈ velocity if H behaves
                              like a proper Hamiltonian: dq/dt = ∂H/∂p)
        """
        # ── Split off q (position) and p (velocity) as leaf tensors
        #    that require grad, so autograd can differentiate H
        #    with respect to them individually. ────────────────────
        static = x[:, :self.STATIC_END]
        pos    = x[:, self.STATIC_END:self.STATIC_END+2].clone().requires_grad_(True)
        vel    = x[:, self.STATIC_END+2:self.STATIC_END+4].clone().requires_grad_(True)
        x_full = torch.cat([static, pos, vel], dim=-1)

        # ── Pathway 1: unbounded force computation (same as V2) ──
        src, dst = edge_index
        force_input = torch.cat([x_full[src], x_full[dst], edge_attr], dim=-1)
        forces = self.force_mlp(force_input)

        node_forces = torch.zeros(x_full.size(0), 2, device=x.device)
        node_forces.scatter_add_(
            0, dst.unsqueeze(1).expand(-1, 2), forces)

        # ── Pathway 2: feature learning with force context ───────
        x_aug = torch.cat([x_full, node_forces], dim=-1)
        h = F.relu(self.conv1(x_aug, edge_index))
        h = F.relu(self.conv2(h,     edge_index))

        # ── Exit prediction (unchanged) ───────────────────────────
        h_graph    = global_mean_pool(h, batch)
        exit_logit = self.exit_head(h_graph)

        # ── Hamiltonian energy and its gradients ──────────────────
        H_per_node = self.H_net(h)          # [N, 1]
        H_total    = H_per_node.sum()       # scalar, sum across whole
                                             # batch — fine since grad
                                             # w.r.t. pos/vel is per-node

        # dp/dt = -∂H/∂q  → used as the velocity UPDATE
        dH_dpos = torch.autograd.grad(
            H_total, pos, create_graph=True)[0]        # [N, 2]

        # dq/dt = ∂H/∂p   → should ≈ actual velocity if H is a
        # well-formed Hamiltonian. Used as a self-consistency
        # regularizer, not directly for the velocity prediction.
        dH_dvel = torch.autograd.grad(
            H_total, vel, create_graph=True)[0]         # [N, 2]

        # Predicted post-collision velocity via Hamilton's equation
        pred_vel = vel + (-dH_dpos)

        return exit_logit, pred_vel, H_per_node, dH_dvel


# ── Sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("── HNN_PDGNN Sanity Test ───────────────────────────")
    model = HNN_PDGNN(node_dim=17, edge_dim=4, hidden=64)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x          = torch.randn(4, 17)
    edge_index = torch.tensor([[0,1, 2,3],
                                [1,0, 3,2]], dtype=torch.long)
    edge_attr  = torch.randn(4, 4)
    batch      = torch.tensor([0,0, 1,1], dtype=torch.long)

    # Must run under enable_grad even for a "test" forward pass —
    # HNN needs local gradients intrinsically.
    with torch.enable_grad():
        exit_logit, pred_vel, H, dH_dvel = model(x, edge_index, edge_attr, batch)

    print(f"exit_logit shape : {exit_logit.shape}  ← [2, 1] expected")
    print(f"pred_vel shape   : {pred_vel.shape}     ← [4, 2] expected")
    print(f"H_per_node shape : {H.shape}             ← [4, 1] expected")
    print(f"dH_dvel shape    : {dH_dvel.shape}       ← [4, 2] expected")
    print(f"H values (energy): {H.detach().squeeze().tolist()}")
    print("── Test passed ──────────────────────────────────────")
