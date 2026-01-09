from flask import Flask, jsonify, send_from_directory, render_template
import os
import glob
import json

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

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np

@app.route('/api/results/<filename>')
def get_result(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
        
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Check if we need to augment with clusters on-the-fly
    if "points" in data and data["points"]:
        sample = data["points"][0]
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

    # Load Acceleration Outliers
    accel_outliers_path = os.path.join(DATA_DIR, "accel_outliers_sample_ids.txt")
    accel_outliers = set()
    if os.path.exists(accel_outliers_path):
        with open(accel_outliers_path, 'r') as f:
            accel_outliers = {line.strip() for line in f if line.strip()}
    
    # Mark acceleration outliers
    if "points" in data:
        for p in data["points"]:
            p["is_accel_outlier"] = p["id"] in accel_outliers
                    
    return jsonify(data)

@app.route('/video/<id>')
def get_video(id):
    # Construct filename: UUID.camera_front_wide_120fov.mp4
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
    if os.path.exists(os.path.join(strategy_dir, filename)):
        return send_from_directory(strategy_dir, filename)
    return "No text description available.", 404

if __name__ == '__main__':
    print(f"Starting viewer on http://localhost:8081")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Results Dir: {RESULTS_DIR}")
    app.run(debug=True, port=8081, host='0.0.0.0')
