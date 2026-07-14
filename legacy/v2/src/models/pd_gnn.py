# src/models/pd_gnn.py
# ─────────────────────────────────────────────────────────────
# Physics-Decoupled Graph Neural Network (PD-GNN)
#
# WHY NOT GATConv:
#   GAT uses softmax to normalize attention weights → bounded [0,1].
#   Physical forces are unbounded vectors. A mass ratio of 10:1
#   should produce a force multiplier of ~10, not 0.9.
#   GATConv fundamentally misrepresents high-energy collisions.
#
# HOW PD-GNN WORKS:
#   Two parallel computation pathways:
#
#   Pathway 1 — Force MLP (UNBOUNDED):
#     Takes src node + dst node + edge attributes.
#     Outputs a 2D force vector (Fx, Fy).
#     NO softmax. NO normalization. Forces can be any magnitude.
#
#   Pathway 2 — Feature GCN (normalized):
#     Standard GCNConv for learning object state representations.
#     Takes node features augmented with computed force vectors.
#     Normalization is fine here — it applies to categorical/positional
#     features, not physical magnitudes.
#
#   The force vectors from Pathway 1 are scattered to destination
#   nodes and appended to node features before Pathway 2 runs.
#   This gives the GCN direct access to the unbounded physical
#   momentum transfer value per node.
#
# ALSO CONTAINS:
#   CausalGNN_V1  — original V1 model (kept for comparison)
#   PD_GNN        — new V2 model
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# ── V1 model (kept for reference and comparison runs) ─────────
class CausalGNN(nn.Module):
    """
    Original V1 GCNConv model.
    node_dim = 17, no edge attributes.
    Kept for baseline comparison during V2 evaluation.
    """
    def __init__(self, in_channels=17, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # edge_attr ignored — V1 compatibility
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ── V2 model — Physics-Decoupled GNN ──────────────────────────
class PD_GNN(nn.Module):
    """
    Physics-Decoupled Graph Neural Network.

    Parameters
    ----------
    node_dim  : input node feature dimension (17 for V2)
    edge_dim  : edge attribute dimension (4 for V2)
    hidden    : hidden layer width
    """

    def __init__(self, node_dim=17, edge_dim=4, hidden=64):
        super().__init__()

        # ── Pathway 1: Force MLP (unbounded output) ──────────
        # Input: [src_features | dst_features | edge_attributes]
        # Output: 2D force vector (Fx, Fy) — NO activation cap
        force_in = node_dim * 2 + edge_dim
        self.force_mlp = nn.Sequential(
            nn.Linear(force_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),   # (Fx, Fy) — unbounded
        )

        # ── Pathway 2: Feature GCN (normalized) ───────────────
        # Receives node features AUGMENTED with force vectors (+2)
        augmented_dim = node_dim + 2
        self.conv1 = GCNConv(augmented_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

        # ── Output heads ──────────────────────────────────────
        # Head 1: Binary exit prediction (primary task)
        self.exit_head = nn.Linear(hidden, 1)

        # Head 2: Post-collision velocity prediction (for Lagrange loss)
        # Predicts (vx, vy) per node — shapes match pre_vel
        self.velocity_head = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Parameters
        ----------
        x          : [N, node_dim]   node features
        edge_index : [2, E]          directed edges
        edge_attr  : [E, edge_dim]   edge attributes
        batch      : [N]             batch assignment vector

        Returns
        -------
        exit_logit : [B, 1]          raw logit for exit prediction
        pred_vel   : [N, 2]          predicted post-collision velocity
        """
        src, dst = edge_index   # source and destination node indices

        # ── Pathway 1: Compute unbounded forces ───────────────
        # Concatenate: source features + destination features + edge attrs
        force_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        forces      = self.force_mlp(force_input)  # [E, 2] — unbounded!

        # Scatter forces to destination nodes (sum aggregation)
        # Each node accumulates all incoming force vectors
        node_forces = torch.zeros(x.size(0), 2, device=x.device)
        node_forces.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, 2),   # broadcast dst indices to 2D
            forces
        )

        # ── Pathway 2: Feature learning with force context ────
        # Augment node features with the computed force vectors
        x_aug = torch.cat([x, node_forces], dim=-1)   # [N, node_dim+2]

        h = F.relu(self.conv1(x_aug, edge_index))
        h = F.relu(self.conv2(h,     edge_index))

        # ── Graph-level readout ───────────────────────────────
        h_graph = global_mean_pool(h, batch)   # [B, hidden]

        # ── Outputs ───────────────────────────────────────────
        exit_logit = self.exit_head(h_graph)   # [B, 1]
        pred_vel   = self.velocity_head(h)     # [N, 2] — per-node velocity

        return exit_logit, pred_vel

    def predict(self, x, edge_index, edge_attr, batch):
        """
        Convenience method: returns just exit probability (no grad).
        Use this in counterfactual.py and benchmark.py.
        """
        self.eval()
        with torch.no_grad():
            logit, _ = self.forward(x, edge_index, edge_attr, batch)
            return torch.sigmoid(logit)


# ── Quick sanity test ─────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print("── PD_GNN Sanity Test ──────────────────────────────")
    model = PD_GNN(node_dim=17, edge_dim=4, hidden=64)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fake batch: 4 nodes (2 graphs × 2 nodes each), 4 directed edges
    x          = torch.randn(4, 17)
    edge_index = torch.tensor([[0,1, 2,3],
                                [1,0, 3,2]], dtype=torch.long)
    edge_attr  = torch.randn(4, 4)
    batch      = torch.tensor([0,0, 1,1], dtype=torch.long)

    logit, vel = model(x, edge_index, edge_attr, batch)

    print(f"exit_logit shape : {logit.shape}   ← [2, 1] expected")
    print(f"pred_vel shape   : {vel.shape}      ← [4, 2] expected")
    print(f"exit probs       : {torch.sigmoid(logit).squeeze().tolist()}")
    print("── Test passed ──────────────────────────────────────")

    print("\n── CausalGNN (V1) Sanity Test ──────────────────────")
    v1 = CausalGNN()
    out = v1(x, edge_index, batch)
    print(f"V1 output shape  : {out.shape}   ← [2, 1] expected")
    print("── V1 test passed ───────────────────────────────────")