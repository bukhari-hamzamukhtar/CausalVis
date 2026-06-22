import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class CausalGNN(torch.nn.Module):
    # Bumped hidden channels to 64 to give the model a bigger brain to learn physics
    def __init__(self, in_channels=17, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        # RIPPED OUT THE SIGMOID! BCEWithLogitsLoss needs raw, un-squashed numbers (logits)
        return self.lin(x)

if __name__ == "__main__":
    # Training script should only run only when executed directly.
    # It must not execute when this module is imported.

    # Make sure you do NOT oversample your dataset!
    # Use the natural split: train_dataset = train_pos + train_neg

    # Calculate the weight for the rare positive cases
    # Since we had roughly 42,876 Negs and 6,002 Pos overall...
    # The ratio in our training set will be around 7.0
    num_negatives = len(train_neg)
    num_positives = len(train_pos)
    pos_weight_val = num_negatives / num_positives
    pos_weight = torch.tensor([pos_weight_val])

    model = CausalGNN()

    # Dropped Learning Rate to 0.005 so it doesn't overshoot the minimum.
    # Added weight_decay (L2 regularization) to prevent it from just memorizing the data.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Bumped to 50 epochs since we slowed down the learning rate
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)

            # Swapped to BCEWithLogitsLoss and plugged in the pos_weight
            loss = F.binary_cross_entropy_with_logits(
                out.view(-1),
                batch.y,
                pos_weight=pos_weight.to(out.device)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}")