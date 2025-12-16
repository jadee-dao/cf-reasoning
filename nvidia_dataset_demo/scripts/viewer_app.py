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

@app.route('/api/results/<filename>')
def get_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

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
    print(f"Starting viewer on http://localhost:8080")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Results Dir: {RESULTS_DIR}")
    app.run(debug=True, port=8080, host='0.0.0.0')
