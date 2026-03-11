from flask import Flask, jsonify, send_from_directory, render_template, request
import os
import json
import glob

app = Flask(__name__)

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
EXTRACTED_DATA_DIR = os.path.join(PROJECT_ROOT, "extracted_data/nvidia_demo")
OBJECT_GRAPHS_DIR = os.path.join(EXTRACTED_DATA_DIR, "object_graphs")
CALIBRATION_DIR = os.path.join(EXTRACTED_DATA_DIR, "calibration")
SAMPLES_DIR = os.path.join(EXTRACTED_DATA_DIR, "samples")
STARRED_FILE = os.path.join(EXTRACTED_DATA_DIR, "starred_samples.json")

# Indexing shards
SHARDS = ["shard_00096", "shard_00097", "shard_00098"]

def load_starred():
    if os.path.exists(STARRED_FILE):
        try:
            with open(STARRED_FILE, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_starred(starred_set):
    with open(STARRED_FILE, 'w') as f:
        json.dump(list(starred_set), f)

def get_shard_index():
    index = []
    starred_set = load_starred()
    for shard in SHARDS:
        jsonl_path = os.path.join(CALIBRATION_DIR, f"{shard}.jsonl")
        if not os.path.exists(jsonl_path):
            continue
            
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    scene_id = data.get("scene_id")
                    timestamp = data.get("timestamp_us")
                    ade = data.get("ADE")
                    if scene_id and timestamp:
                        sample_id = f"{scene_id}_{timestamp}"
                        index.append({
                            "shard": shard,
                            "scene_id": scene_id,
                            "timestamp": timestamp,
                            "sample_id": sample_id,
                            "ade": ade,
                            "starred": sample_id in starred_set
                        })
                except:
                    continue
    return index

@app.route('/')
def index():
    return render_template('shard_viewer.html')

@app.route('/api/nvidia/list')
def list_samples():
    index = get_shard_index()
    return jsonify(index)

@app.route('/api/nvidia/starred')
def list_starred():
    return jsonify(list(load_starred()))

@app.route('/api/nvidia/star', methods=['POST'])
def toggle_star():
    data = request.json
    sample_id = data.get('sample_id')
    starred = data.get('starred') # boolean
    
    if not sample_id:
        return "Missing sample_id", 400
        
    starred_set = load_starred()
    if starred:
        starred_set.add(sample_id)
    else:
        if sample_id in starred_set:
            starred_set.remove(sample_id)
            
    save_starred(starred_set)
    return jsonify({"status": "success", "sample_id": sample_id, "starred": starred})

@app.route('/api/nvidia/data/<shard>/<scene_id>/<timestamp>')
def get_sample_data(shard, scene_id, timestamp):
    sample_id = f"{scene_id}_{timestamp}"
    
    # 1. Get metrics and waypoints from JSONL
    jsonl_path = os.path.join(CALIBRATION_DIR, f"{shard}.jsonl")
    sample_info = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get("scene_id") == scene_id and data.get("timestamp_us") == int(timestamp):
                    sample_info = data
                    break
    
    # 2. Get Object Graph
    graph_path = os.path.join(OBJECT_GRAPHS_DIR, shard, f"{sample_id}_graph.json")
    object_graph = {}
    if os.path.exists(graph_path):
        with open(graph_path, 'r') as f:
            object_graph = json.load(f)
            
    return jsonify({
        "info": sample_info,
        "graph": object_graph
    })

@app.route('/api/nvidia/image/<sample_id>')
def get_image(sample_id):
    # Images are flat in samples directory (or might be in shard subdirs if they were extracted that way)
    # Check both
    filename = f"{sample_id}.jpg"
    if os.path.exists(os.path.join(SAMPLES_DIR, filename)):
        return send_from_directory(SAMPLES_DIR, filename)
    
    # Check shard subdirs
    for shard in SHARDS:
        shard_img_path = os.path.join(SAMPLES_DIR, shard, filename)
        if os.path.exists(shard_img_path):
            return send_from_directory(os.path.dirname(shard_img_path), filename)
            
    return "Image not found", 404

if __name__ == '__main__':
    print(f"Starting Shard Viewer on http://localhost:8082")
    app.run(debug=True, port=8082, host='0.0.0.0')
