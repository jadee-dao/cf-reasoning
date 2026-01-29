from flask import Flask, jsonify, send_from_directory, render_template, request
import os
import glob
import json

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# analysis_results is local to scripts/
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")
# extracted_data is ../extracted_data
DATA_ROOT = os.path.join(BASE_DIR, "../extracted_data")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/list')
def list_results():
    # Return structure: { "strategies": [...], "datasets": { "strategy": [d1, d2] } }
    base_proj = os.path.join(RESULTS_DIR, "projections")
    strategies = []
    structure = {}
    
    if os.path.exists(base_proj):
        # List strategies (folders)
        for strat in os.listdir(base_proj):
            strat_path = os.path.join(base_proj, strat)
            if os.path.isdir(strat_path):
                strategies.append(strat)
                structure[strat] = []
                # List datasets (json files)
                for f in os.listdir(strat_path):
                    if f.endswith(".json"):
                        dataset = os.path.splitext(f)[0]
                        structure[strat].append(dataset)
                structure[strat].sort()
                
    strategies.sort()
    return jsonify({
        "strategies": strategies,
        "structure": structure
    })

@app.route('/api/results')
def get_result():
    # Query params: strategy, datasets (comma-separated), filter_same_id, filter_diff_dataset
    strategy = request.args.get('strategy')
    datasets_str = request.args.get('datasets')
    filter_same = request.args.get('filter_same_id', 'false') == 'true'
    filter_diff_dataset = request.args.get('filter_diff_dataset', 'false') == 'true'
    
    if not strategy or not datasets_str:
        return "Missing strategy or datasets parameter", 400
        
    datasets = datasets_str.split(',')
    
    all_points = []
    all_pairs = []
    
    # Lookup: id -> dataset
    id_to_dataset = {}
    
    # 1. Load Independent Projections (Points)
    for dataset in datasets:
        dataset = dataset.strip()
        if not dataset: continue
        if dataset == "global": continue # specific check to avoid loading global as independent
        
        proj_path = os.path.join(RESULTS_DIR, "projections", strategy, f"{dataset}.json")
        if os.path.exists(proj_path):
            with open(proj_path, 'r') as f:
                data = json.load(f) # {"points": [...]}
                for p in data.get("points", []):
                    if "dataset" not in p:
                        p["dataset"] = dataset 
                    id_to_dataset[p["id"]] = p["dataset"]
                    
                    # Rename 'projections' to 'projections_independent'
                    if "projections" in p:
                        p["projections_independent"] = p.pop("projections")
                        
                    all_points.append(p)

    # 2. Load Global Projections and Merge
    global_path = os.path.join(RESULTS_DIR, "projections", strategy, "global.json")
    if os.path.exists(global_path) and all_points:
        # Build lookup for currently loaded points to avoid iterating full global file if huge? 
        # Actually we have to read global file anyway.
        # Let's read global file and create a map for RELEVANT IDs.
        
        # Optimization: Set of loaded IDs
        loaded_ids = set(p["id"] for p in all_points)
        
        with open(global_path, 'r') as f:
            g_data = json.load(f)
            for gp in g_data.get("points", []):
                if gp["id"] in loaded_ids:
                    # Find the point object in all_points (need a map or linear scan? linear is slow)
                    # Let's map all_points by ID first
                    pass 
                    
        # Map all_points by ID for fast merge
        points_map = {p["id"]: p for p in all_points}
        
        with open(global_path, 'r') as f:
            g_data = json.load(f)
            for gp in g_data.get("points", []):
                if gp["id"] in points_map:
                    # Attach global projections
                    points_map[gp["id"]]["projections_global"] = gp.get("projections", {})
                    # Also merge clusters/outliers from global if needed? 
                    # Usually global outlier/clusters are what we want for "global" view.
                    # They are inside 'projections' usually in our structure (mixed in).
                    # Wait, compute_projections structure:
                    # point_data["projections"][method] = {x, y, is_outlier, cluster_k3...}
                    # So copying "projections" is correct.

    # 3. Load Similarities (Pairs)
    for dataset in datasets:
        dataset = dataset.strip()
        if not dataset: continue
        if dataset == "global": continue
        
        sim_path = os.path.join(RESULTS_DIR, "similarities", strategy, f"{dataset}.json")
        if os.path.exists(sim_path):
            with open(sim_path, 'r') as f:
                sim_dict = json.load(f)
                
                # Helper to process items
                def process_item(item_pair, item_score):
                    id1, id2 = item_pair
                    
                    # Filter: Same Scene (using rsplit to be robust)
                    if filter_same:
                        # Split on LAST underscore: scene-123_456 -> scene-123
                        base1 = id1.rsplit('_', 1)[0]
                        base2 = id2.rsplit('_', 1)[0]
                        if base1 == base2:
                            return None
                            
                    # Filter: Different Datasets Only
                    if filter_diff_dataset:
                        d1 = id_to_dataset.get(id1)
                        d2 = id_to_dataset.get(id2)
                        # If either is unknown, keep it? Or skip? Let's skip if both known and equal.
                        if d1 and d2 and d1 == d2:
                            return None
                            
                    return {
                        "pair": [id1, id2],
                        "score": float(item_score),
                        "dataset": dataset
                    }

                # Handle dict or list format
                if isinstance(sim_dict, dict):
                    for id1, targets in sim_dict.items():
                        for id2, score in targets.items():
                            res = process_item([id1, id2], score)
                            if res: all_pairs.append(res)
                elif isinstance(sim_dict, list):
                    for item in sim_dict:
                        res = process_item(item["pair"], item["score"])
                        if res: all_pairs.append(res)

    # Sort entire list of pairs descending
    all_pairs.sort(key=lambda x: x["score"], reverse=True)
    
    # Global Stats
    max_score = all_pairs[0]["score"] if all_pairs else 0
    min_score = all_pairs[-1]["score"] if all_pairs else 0
    
    response = {
        "strategy": strategy,
        "datasets": datasets,
        "points": all_points,
        "all_pairs": all_pairs[:200], # Top 200
        "least_similar": all_pairs[-20:][::-1] if len(all_pairs) > 20 else all_pairs[::-1][:20], # Bottom 20
        "total_pairs": len(all_pairs),
        "global_max_score": max_score,
        "global_min_score": min_score
    }
    
    return jsonify(response)


@app.route('/video/<path:id>')
def get_video(id):
    # ID might be just UUID_TIMESTAMP or maybe have dataset?
    # We search extracted_data/*/samples/{id}.mp4
    
    # Security check: id shouldn't have ..
    if ".." in id:
         return "Invalid ID", 400
         
    # Try to find the file
    # Pattern: extracted_data/*/samples/{id}.mp4
    pattern = os.path.join(DATA_ROOT, "*", "samples", f"{id}.mp4")
    matches = glob.glob(pattern)
    
    if matches:
        # Take first match
        path = matches[0]
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        return send_from_directory(directory, filename)
        
    return "Video not found", 404

@app.route('/debug/<path:selection>/<id>')
def get_debug_image(selection, id):
    # selection is "STRATEGY/DATASET"
    if "/" not in selection:
        return "Invalid selection", 400
        
    strategy, dataset = selection.split("/", 1)
    
    # Path: analysis_results/visualize_samples/{strategy}/{dataset}/{id}.jpg
    img_dir = os.path.join(RESULTS_DIR, "visualize_samples", strategy, dataset)
    filename = f"{id}.jpg"
    
    if os.path.exists(os.path.join(img_dir, filename)):
        return send_from_directory(img_dir, filename)
        
    return "Debug image not found", 404

@app.route('/debug_text/<path:selection>/<id>')
def get_debug_text(selection, id):
    # selection is "STRATEGY/DATASET"
    if "/" not in selection:
        return "Invalid selection", 400
        
    strategy, dataset = selection.split("/", 1)
    
    # Path: analysis_results/visualize_samples/{strategy}/{dataset}/{id}.txt
    txt_dir = os.path.join(RESULTS_DIR, "visualize_samples", strategy, dataset)
    filename = f"{id}.txt"
    filepath = os.path.join(txt_dir, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
        
    return "", 404

if __name__ == '__main__':
    print(f"Starting viewer on http://localhost:8081")
    app.run(debug=True, port=8081, host='0.0.0.0')
