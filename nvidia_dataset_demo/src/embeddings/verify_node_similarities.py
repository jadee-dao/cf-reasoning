import json
import numpy as np
import os

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_similarities(json_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    # Filter out ego node if it has zero features
    obj_nodes = [n for n in nodes if n['type'] == 'object']
    
    if len(obj_nodes) < 2:
        print("Not enough object nodes to compare.")
        return

    print(f"Comparing {len(obj_nodes)} object nodes...")
    print("-" * 60)
    print(f"{'Node A':<15} | {'Node B':<15} | {'Similarity':<10}")
    print("-" * 60)

    for i in range(len(obj_nodes)):
        for j in range(i + 1, len(obj_nodes)):
            n_a = obj_nodes[i]
            n_b = obj_nodes[j]
            
            sim = cosine_similarity(n_a['features'], n_b['features'])
            
            label_a = f"{n_a['id']} ({n_a['label']})"
            label_b = f"{n_b['id']} ({n_b['label']})"
            
            print(f"{label_a:<15} | {label_b:<15} | {sim:.4f}")

    # Calculate average similarity within same class vs different class
    classes = {}
    for n in obj_nodes:
        cls = n['label']
        if cls not in classes:
            classes[cls] = []
        classes[cls].append(n['features'])

    print("\nClass-wise Average Similarities:")
    labels = list(classes.keys())
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            l_a = labels[i]
            l_b = labels[j]
            
            sims = []
            feats_a = classes[l_a]
            feats_b = classes[l_b]
            
            for f_a in feats_a:
                for f_b in feats_b:
                    # Avoid self-comparison if same class
                    if np.array_equal(f_a, f_b):
                        continue
                    
                    s = cosine_similarity(f_a, f_b)
                    # Skip effectively identical points (patch artifacts) for semantic average
                    if s < 0.9999:
                        sims.append(s)
            
            if sims:
                avg_sim = np.mean(sims)
                print(f"{l_a} vs {l_b}: {avg_sim:.4f}")

if __name__ == "__main__":
    json_path = "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/test_outputs/graph_vis_graph.json"
    verify_similarities(json_path)
