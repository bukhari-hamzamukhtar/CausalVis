import networkx as nx

def build_collision_subgraph(objects, collisions, positions, collision_idx, window=5):
    """Build one subgraph per collision event."""
    event = collisions[collision_idx]
    frame = event['frame']
    obj_ids = event['object']
    
    G = nx.Graph()
    G.graph['collision_frame'] = frame
    
    for oid in obj_ids:
        obj = objects[oid]
        key = f"{obj['color']}_{obj['material']}_{obj['shape']}"
        
        # Get position trajectory across window
        traj = []
        for f in range(max(0, frame - window), frame + window + 1):
            if f in positions and key in positions[f]:
                traj.append(positions[f][key])
        
        G.add_node(oid, color=obj['color'], material=obj['material'],
                    shape=obj['shape'], trajectory=traj)
    
    # Edge between colliding objects
    if len(obj_ids) == 2:
        G.add_edge(obj_ids[0], obj_ids[1], frame=frame)
    
    return G

if __name__ == "__main__":
    from loader import load_scene
    objects, collisions, positions = load_scene(r"G:\CausalVis\data\processed_proposals\sim_00000.json")
    
    for i in range(len(collisions)):
        g = build_collision_subgraph(objects, collisions, positions, i)
        print(f"Subgraph {i}: nodes={list(g.nodes())}, frame={g.graph['collision_frame']}")
