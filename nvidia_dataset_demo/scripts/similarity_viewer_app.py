
from flask import Flask, jsonify, send_from_directory, render_template
import os
import glob
import json

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data dir is parallel to scripts
DATA_DIR = os.path.join(BASE_DIR, "../extracted_data")
# Results in scripts/analysis_results/similarity_plots
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results/similarity_plots")

@app.route('/')
def index():
    return render_template('similarity_viewer.html')

@app.route('/api/list')
def list_results():
    results = []
    seen_ids = set()

    for f in glob.glob(os.path.join(RESULTS_DIR, "*_results.json")):
        sid = os.path.basename(f).replace("_results.json", "")
        if sid in seen_ids: continue
        seen_ids.add(sid)
        
        is_outlier = False
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                is_outlier = data.get('is_outlier', False)
        except:
            pass
            
        results.append({
            "id": sid,
            "is_outlier": is_outlier,
            "legacy": False
        })
        
    for f in glob.glob(os.path.join(RESULTS_DIR, "*_similarity_plot.png")):
        sid = os.path.basename(f).replace("_similarity_plot.png", "")
        if sid not in seen_ids:
            seen_ids.add(sid)
            results.append({"id": sid, "is_outlier": True, "legacy": True})

    for f in glob.glob(os.path.join(RESULTS_DIR, "*_comparison_plot.png")):
        sid = os.path.basename(f).replace("_comparison_plot.png", "")
        if sid not in seen_ids:
            seen_ids.add(sid)
            results.append({"id": sid, "is_outlier": True, "legacy": True})
        
    results.sort(key=lambda x: x['id'])
    return jsonify(results)

@app.route('/api/data/<sample_id>')
def get_data(sample_id):
    # Try modern JSON first
    json_path = os.path.join(RESULTS_DIR, f"{sample_id}_results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    
    # Fallback to legacy
    # We construct a "dummy" data object that tells frontend to use static image
    return jsonify({
        "legacy": True,
        "sample_id": sample_id
    })

@app.route('/plot_image/<sample_id>')
def get_plot_image(sample_id):
    # Try comparison plot first (multi-strategy)
    p1 = f"{sample_id}_comparison_plot.png"
    if os.path.exists(os.path.join(RESULTS_DIR, p1)):
        return send_from_directory(RESULTS_DIR, p1)
        
    # Legacy plot
    p2 = f"{sample_id}_similarity_plot.png"
    if os.path.exists(os.path.join(RESULTS_DIR, p2)):
        return send_from_directory(RESULTS_DIR, p2)
        
    return "Not found", 404

@app.route('/video/<sample_id>')
def get_video(sample_id):
    # Try sampled video (clean)
    v1 = f"{sample_id}_sampled.mp4"
    if os.path.exists(os.path.join(RESULTS_DIR, v1)):
        return send_from_directory(RESULTS_DIR, v1)
        
    # Legacy video (annotated)
    v2 = f"{sample_id}_similarity.mp4"
    if os.path.exists(os.path.join(RESULTS_DIR, v2)):
        return send_from_directory(RESULTS_DIR, v2)
        
    return "Not found", 404

if __name__ == '__main__':
    print(f"Starting Similarity Viewer on http://localhost:8080")
    print(f"Results Dir: {RESULTS_DIR}")
    app.run(debug=True, port=8080, host='0.0.0.0')
