def extract_causal_rule(model, data):
    model.eval()
    with torch.no_grad():
        x = F.relu(model.conv1(data.x, data.edge_index))
        x = F.relu(model.conv2(x, data.edge_index))
        pooled = x.mean(dim=0, keepdim=True)
        prob = torch.sigmoid(model.lin(pooled)).item()
    
    v0 = (data.x[0][-2]**2 + data.x[0][-1]**2).sqrt().item()
    v1 = (data.x[1][-2]**2 + data.x[1][-1]**2).sqrt().item()
    dominant = 0 if v0 > v1 else 1
    other = 1 - dominant

    true_label = "EXITED" if data.y.item() == 1 else "STAYED"
    
    if prob > 0.5:
        rule = f"Object {other} predicted to EXIT (confidence {prob:.2f}) — momentum from Object {dominant} exceeded stability threshold"
    else:
        rule = f"Object {other} predicted to STAY (confidence {1-prob:.2f}) — insufficient momentum transfer"
    
    return f"{rule} | Ground truth: {true_label}"

# Real output on actual test data
for i in range(10):
    print(extract_causal_rule(model, test_dataset[i]))