from flask import Flask, jsonify, send_from_directory, render_template
import os
import glob
import json
import csv
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../extracted_data")
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")
DEBUG_DIR = os.path.join(RESULTS_DIR, "debug_images")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/list')
def list_results():
    pattern = os.path.join(RESULTS_DIR, "results_*.json")
    files = glob.glob(pattern)
    filenames = [os.path.basename(f) for f in files]
    return jsonify(filenames)

@app.route('/api/results/<filename>')
def get_result(filename):
    try:
        file_path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(file_path):
            return "File not found", 404
            
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Attempt to load corresponding CSV projections
        strategy_name = filename.replace("results_", "").replace(".json", "")
        csv_filename = f"projections_{strategy_name}.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_filename)

        if os.path.exists(csv_path):
            print(f"Merging external projections from {csv_filename}...")
            csv_data = {}
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_data[row['id']] = row
            
            if "points" in data:
                for p in data["points"]:
                    pid = p["id"]
                    if pid in csv_data:
                        row = csv_data[pid]
                        try:
                            # Update projection fields using .get with default '0' strings for safety
                            p["pca"] = [float(row.get('pca_1', '0')), float(row.get('pca_2', '0'))]
                            if 'tsne_1' in row and row['tsne_1']:
                                p["tsne"] = [float(row['tsne_1']), float(row['tsne_2'])]
                            if 'umap_1' in row and row['umap_1']:
                                p["umap"] = [float(row['umap_1']), float(row['umap_2'])]
                            
                            # Update cluster fields
                            for k in [3, 5, 8]:
                                key = f"cluster_k{k}"
                                if key in row and row[key]:
                                    p[key] = int(row[key])
                            
                            # Update outlier status
                            if "is_outlier" in row:
                                # CSV: -1 is outlier, 1 is inlier. Map -1 to True.
                                val = int(row["is_outlier"])
                                p["is_outlier"] = (val == -1)
                        except (ValueError, TypeError) as e:
                            # Log but accumulate successfully parsed parts, or skip this point update?
                            # For now, print error and continue with what we have
                            print(f"Error parsing CSV data for {pid}: {e}")
            
        # Check if we need to augment with clusters on-the-fly
        if "points" in data and data["points"]:
            sample = data["points"][0]
            # Only compute if missing and if no external CSV provided them (though CSV logic above should have filled them)
            # The previous logic checked 'cluster_k3' not in sample.
            # If CSV merge happened, sample has it. If not, we compute.
            if "cluster_k3" not in sample:
                print(f"Augmenting {filename} with clusters on-the-fly...")
                
                # Extract coordinates for clustering
                # Prefer tsne, then pca, then just x/y
                coords = []
                for p in data["points"]:
                    if "tsne" in p:
                        coords.append(p["tsne"])
                    elif "pca" in p:
                        coords.append(p["pca"])
                    else:
                        coords.append([p.get("x", 0), p.get("y", 0)])
                
                X = np.array(coords)
                n_samples = len(X)
                
                # 1. Outliers
                iso = IsolationForest(contamination='auto', random_state=42)
                outliers = iso.fit_predict(X)
                
                # 2. Clusters
                k_values = [3, 5, 8]
                clusters = {}
                for k in k_values:
                    if n_samples >= k:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        clusters[k] = kmeans.fit_predict(X)
                    else:
                        clusters[k] = np.zeros(n_samples)
                
                # Inject back
                for i, p in enumerate(data["points"]):
                    p["is_outlier"] = bool(outliers[i] == -1)
                    for k in k_values:
                        p[f"cluster_k{k}"] = int(clusters[k][i])

        # Load Acceleration Outliers (from JSON)
        accel_outliers_path = os.path.join(DATA_DIR, "calibration_set/worst-ade-log-10-90pctl.json")
        accel_outliers = set()
        if os.path.exists(accel_outliers_path):
            try:
                with open(accel_outliers_path, 'r') as f:
                    outlier_data = json.load(f)
                    if "results" in outlier_data:
                        for chunk_key, res in outlier_data["results"].items():
                            scene_id = res.get("scene_id")
                            if scene_id and "top3_worst" in res:
                                for item in res["top3_worst"]:
                                    t_val = item.get("t_rel_us")
                                    if t_val is not None:
                                        accel_outliers.add(f"{scene_id}_{t_val}")
            except Exception as e:
                print(f"Error loading outlier JSON: {e}")
        
        # Mark acceleration outliers
        if "points" in data:
            for p in data["points"]:
                p["is_accel_outlier"] = p["id"] in accel_outliers
        
        # Filter and Slice Pairs
        from flask import request
        filter_same_id = request.args.get('filter_same_id', 'false').lower() == 'true'
        
        if "all_pairs" in data:
            pairs = data["all_pairs"]
            
            if filter_same_id:
                filtered_pairs = []
                for p in pairs:
                    # id format: UUID_TIMESTAMP
                    id1 = p["pair"][0]
                    id2 = p["pair"][1]
                    uuid1 = id1.rsplit('_', 1)[0] if '_' in id1 else id1
                    uuid2 = id2.rsplit('_', 1)[0] if '_' in id2 else id2
                    if uuid1 != uuid2:
                        filtered_pairs.append(p)
                pairs = filtered_pairs
            
            # Compute stats BEFORE Slicing
            if pairs:
                data["global_max_score"] = pairs[0]["score"]
                data["global_min_score"] = pairs[-1]["score"]
                data["least_similar"] = pairs[-5:][::-1] # Reverse to have worst first
                data["total_pairs"] = len(pairs) # Update total count based on filter
            else:
                data["global_max_score"] = 0
                data["global_min_score"] = 0
                data["least_similar"] = []
                data["total_pairs"] = 0

            data["all_pairs"] = pairs[:100]

        return jsonify(data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Internal Server Error: {str(e)}", 500

@app.route('/video/<id>')
def get_video(id):
    # Construct filename: UUID.camera_front_wide_120fov.mp4
    # Handle IDs with timestamp suffix (e.g. UUID_TIMESTAMP)
    
    if '_' in id:
        parts = id.rsplit('_', 1)
        if parts[1].isdigit():
            uuid_val = parts[0]
            timestamp = parts[1]
            
            # Check for interval clip in 1.5s directory
            # Path: DATA_DIR/interval_samples/1.5s/UUID/TIMESTAMP.mp4
            interval_dir = os.path.join(DATA_DIR, "interval_samples", "1.5s", uuid_val)
            clip_filename = f"{timestamp}.mp4"
            
            if os.path.exists(os.path.join(interval_dir, clip_filename)):
                return send_from_directory(interval_dir, clip_filename)
                
            # If clip not found, fallback to UUID for full video (though user complained about this)
            id = uuid_val

    filename = f"{id}.camera_front_wide_120fov.mp4"
    return send_from_directory(DATA_DIR, filename)

@app.route('/debug/<strategy>/<id>')
def get_debug_image(strategy, id):
    # Debug images are saved as {id}.jpg inside the strategy folder
    filename = f"{id}.jpg"
    strategy_dir = os.path.join(DEBUG_DIR, strategy)
    return send_from_directory(strategy_dir, filename)

@app.route('/debug/text/<strategy>/<id>')
def get_debug_text(strategy, id):
    # Debug text saved as {id}.txt inside the strategy folder
    filename = f"{id}.txt"
    strategy_dir = os.path.join(DEBUG_DIR, strategy)
    file_path = os.path.join(strategy_dir, filename)
    
    if os.path.exists(file_path):
        # Check content for "failed" string
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if "analysis failed" in content.lower():
                     return "No text description available.", 404
                return content
        except Exception:
            return "Error reading description.", 500
            
    return "No text description available.", 404

if __name__ == '__main__':
    print(f"Starting viewer on http://localhost:8081")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Results Dir: {RESULTS_DIR}")
    app.run(debug=True, port=8081, host='0.0.0.0')
