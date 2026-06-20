import networkx as nx
import pycocotools
import matplotlib.pyplot as plt
import os

def build_video_causal_graph(objects, collisions, positions, window=5):
    """
    Builds a single Directed Graph (DiGraph) representing the entire video timeline.
    Links objects involved in collisions and draws directed temporal tracking edges
    across consecutive collision events.
    """
    G = nx.DiGraph()
    
    # Track the last node ID used for each object type to link them temporally
    last_collision_node = {} # {obj_id: uniquely_labeled_node_in_graph}
    
    for idx, event in enumerate(collisions):
        frame = event['frame']
        obj_ids = event['object']
        
        # 1. Create sub-event collision nodes for this specific frame
        current_event_nodes = []
        for oid in obj_ids:
            obj = objects[oid]
            node_key = f"sub_{idx}_obj_{oid}"
            
            # Extract trajectory
            traj = []
            for f in range(max(0, frame - window), frame + window + 1):
                if f in positions and f"{obj['color']}_{obj['material']}_{obj['shape']}" in positions[f]:
                    traj.append(positions[f][f"{obj['color']}_{obj['material']}_{obj['shape']}"])
            
            # Construct display label
            label = f"{obj['color']}\n{obj['material']}\n{obj['shape']}"
            
            G.add_node(node_key, 
                       orig_id=oid, 
                       color=obj['color'], 
                       material=obj['material'], 
                       shape=obj['shape'], 
                       label=label, 
                       trajectory=traj,
                       collision_idx=idx)
            
            current_event_nodes.append(node_key)
            
            # 2. If this object was in a previous collision, draw the temporal tracking arrow
            if oid in last_collision_node:
                G.add_edge(last_collision_node[oid], node_key, type='temporal_link', label='Carries Momentum')
            
            # Update the history tracker for this object identity
            last_collision_node[oid] = node_key
            
        # 3. Draw directed collision interactions between the participants
        if len(current_event_nodes) == 2:
            # Directed edge representing interaction transfer at that frame boundary
            G.add_edge(current_event_nodes[0], current_event_nodes[1], type='collision', frame=frame, label=f"Collision @ F:{frame}")
            G.add_edge(current_event_nodes[1], current_event_nodes[0], type='collision', frame=frame, label=f"Collision @ F:{frame}")

    return G

if __name__ == "__main__":
    from loader import load_scene
    
    json_path = r"G:\CausalVis\data\processed_proposals\sim_00000.json"
    if os.path.exists(json_path):
        objects, collisions, positions = load_scene(json_path)
        
        # Build the entire interconnected video timeline structure
        full_graph = build_video_causal_graph(objects, collisions, positions)
        
        # Plot setup
        plt.figure(figsize=(14, 8))
        pos = nx.shell_layout(full_graph) # Arranges sub-events in readable groupings
        
        # Colors and labels extraction
        node_colors = [data['color'] for node, data in full_graph.nodes(data=True)]
        node_labels = nx.get_node_attributes(full_graph, 'label')
        
        # Separate edge types for visual distinction
        collision_edges = [(u, v) for u, v, data in full_graph.edges(data=True) if data['type'] == 'collision']
        temporal_edges = [(u, v) for u, v, data in full_graph.edges(data=True) if data['type'] == 'temporal_link']
        
        # Draw Nodes
        nx.draw_networkx_nodes(full_graph, pos, node_color=node_colors, node_size=3500)
        nx.draw_networkx_labels(full_graph, pos, labels=node_labels, font_size=8, font_color='white', font_weight='bold')
        
        # Draw Directed Collision Edges (Solid Red lines with arrows)
        nx.draw_networkx_edges(full_graph, pos, edgelist=collision_edges, edge_color='red', width=2, arrows=True, arrowsize=20)
        
        # Draw Directed Causal Temporal Links (Dashed Blue arrows tracking objects across time)
        nx.draw_networkx_edges(full_graph, pos, edgelist=temporal_edges, edge_color='blue', style='dashed', width=3, arrows=True, arrowsize=25)
        
        # Edge Labels
        collision_labels = {(u, v): data['label'] for u, v, data in full_graph.edges(data=True) if data['type'] == 'collision'}
        temporal_labels = {(u, v): data['label'] for u, v, data in full_graph.edges(data=True) if data['type'] == 'temporal_link'}
        
        nx.draw_networkx_edge_labels(full_graph, pos, edge_labels=collision_labels, font_color='red', font_size=8)
        nx.draw_networkx_edge_labels(full_graph, pos, edge_labels=temporal_labels, font_color='blue', font_size=8)
        
        plt.title("Full Video Timeline Causal Architecture: Collision Nodes Linked via Temporal Identity Edges")
        plt.axis('off')
        plt.show()
    else:
        print("Check path setup.")