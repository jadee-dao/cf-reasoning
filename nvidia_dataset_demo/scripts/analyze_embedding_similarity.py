
import os
import glob
import json
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load .env from repo root (2 levels up from scripts/analyze_embedding_similarity.py?)
# script is in nvidia_dataset_demo/scripts
# .env is in cf-reasoning/.env (so ../../ ?)
# User said @[.env] is /home/jadelynn/cf-reasoning/.env
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
ENV_PATH = os.path.join(REPO_ROOT, ".env")
if os.path.exists(ENV_PATH):
    print(f"Loading .env from {ENV_PATH}")
    load_dotenv(ENV_PATH)
else:
    print(f"Warning: .env not found at {ENV_PATH}")

DATA_DIR = os.path.join(BASE_DIR, "../extracted_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results/similarity_plots")
GT_FILE = os.path.join(DATA_DIR, "accel_outliers_sample_ids.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import Strategies
try:
    from embeddings.strategies import (
        NaiveStrategy, 
        TextDescriptionStrategy, 
        ForegroundStrictStrategy, 
        VLMCaptionStrategy,
        # FastVLMHazardStrategy,
        FastViTAttentionStrategy,
        OpenRouterHazardStrategy
    )
except ImportError:
    import sys
    sys.path.append(BASE_DIR)
    from embeddings.strategies import (
        NaiveStrategy, 
        TextDescriptionStrategy, 
        ForegroundStrictStrategy, 
        VLMCaptionStrategy,
        # FastVLMHazardStrategy,
        FastViTAttentionStrategy,
        OpenRouterHazardStrategy
    )

# ... (rest of code)

    # 5. OpenRouter Hazard (Replaces FastVLM)
    print("  Loading OpenRouterHazardStrategy...")
    try:
        s = OpenRouterHazardStrategy()
        # OpenRouter strategy checks for API key in load_model/init
        # We can check here to avoid later crash or just let it handle it
        if not os.getenv("OPENROUTER_API_KEY"):
            print("    [WARNING] OPENROUTER_API_KEY not found. Skipping OpenRouter Hazard.")
        else:
            s.load_model()
            strategies['Hazard'] = s
    except Exception as e:
        print(f"    [Error] Could not load OpenRouter Strategy: {e}")
    
    # 6. FastViT Attention (Disabled)
    # print("  Loading FastViTAttentionStrategy...")
    # s = FastViTAttentionStrategy()
    # s.load_model()
    # strategies['ViT Attention'] = s

def load_gt_ids(filepath):
    if not os.path.exists(filepath):
        print(f"Error: GT file not found at {filepath}")
        return set()
    with open(filepath, 'r') as f:
        ids = {line.strip() for line in f if line.strip()}
    return ids

def get_sample_paths(sample_id):
    video_path = glob.glob(os.path.join(DATA_DIR, f"{sample_id}.*.mp4"))
    ts_path = glob.glob(os.path.join(DATA_DIR, f"{sample_id}.*.timestamps.parquet"))
    
    if not video_path or not ts_path:
        return None, None
        
    return video_path[0], ts_path[0]

def get_status(score):
    if score >= 0.95:
        return "Stable", (0, 255, 0) # Green
    elif score >= 0.90:
        return "Change", (0, 255, 255) # Yellow
    else:
        return "Dip", (0, 0, 255) # Red

import contextlib
import io

@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield

def analyze_sample(sample_id, strategies, interval_seconds=0.5, regenerate=False, is_outlier=True):
    """
    Analyzes embedding similarity with multiple strategies and creates an HTML viewer.
    """
    video_path, ts_path = get_sample_paths(sample_id)
    if not video_path:
        print(f"Missing files for {sample_id}")
        return

    # Load Timestamps
    df_ts = pd.read_parquet(ts_path)
    if 'timestamp' not in df_ts.columns:
        print(f"No timestamp column in {ts_path}")
        return

    timestamps_us = df_ts['timestamp'].values
    min_ts = timestamps_us.min()
    max_ts = timestamps_us.max()
    interval_us = interval_seconds * 1_000_000
    
    # Generate target timestamps (float)
    target_timestamps_float = np.arange(min_ts, max_ts, interval_us)
    target_timestamps = target_timestamps_float.astype(np.int64)
    
    # Find closest frames
    df_targets = pd.DataFrame({'target_ts': target_timestamps})
    df_ts_sorted = df_ts.sort_values('timestamp').reset_index()
    df_merged = pd.merge_asof(df_targets, df_ts_sorted, left_on='target_ts', right_on='timestamp', direction='nearest')
    
    target_frames_info = df_merged[['frame_index', 'timestamp']].to_dict('records')
    
    # --- Load Existing Results ---
    json_path = os.path.join(OUTPUT_DIR, f"{sample_id}_results.json")
    existing_results = {}
    if os.path.exists(json_path) and not regenerate:
        try:
            with open(json_path, 'r') as f:
                existing_json = json.load(f)
                # Verify length match
                if len(existing_json.get('timestamps', [])) == len(target_frames_info):
                     existing_results = existing_json.get('strategies', {})
                     print(f"Loaded existing results for {sample_id}: {list(existing_results.keys())}")
                else:
                    print(f"Existing results length mismatch ({len(existing_json.get('timestamps', []))} vs {len(target_frames_info)}). Regenerating.")
        except Exception as e:
            print(f"Error loading existing JSON: {e}")

    # Determine which strategies need computing
    strategies_to_compute = {}
    final_strategies_list = list(strategies.keys()) # For order
    
    for name, s in strategies.items():
        if name not in existing_results:
            strategies_to_compute[name] = s
        else:
            pass # We have it
            
    if not strategies_to_compute and not regenerate:
        print(f"Skipping {sample_id} computation (all strategies present). Regenerating video only if needed.")
        # We could skip video gen if not needed, but user wants updated annotations potentially?
        # Let's assume we proceed to video gen reusing scores.
    else:
        print(f"Computing NEW strategies for {sample_id}: {list(strategies_to_compute.keys())}")

    print(f"Analyzing {sample_id}: {len(target_frames_info)} frames (every {interval_seconds}s)")
    
    cap = cv2.VideoCapture(video_path)
    
    # Setup Output Video
    out_video_filename = f"{sample_id}_sampled.mp4"
    out_video_path = os.path.join(OUTPUT_DIR, out_video_filename)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = 1.0 / interval_seconds
    
    # Use 'mp4v' initially, re-encode later
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, out_fps, (width, height))
    
    # Data columns
    prev_embeddings = {name: None for name in strategies_to_compute.keys()}
    
    # Initialize results containers
    # We want 'results' to contain ALL data (old + new)
    results = {name: [] for name in strategies.keys()}
    
    plot_timestamps = []
    
    temp_img_path = os.path.join(OUTPUT_DIR, "temp_frame.jpg")

    for i, info in enumerate(tqdm(target_frames_info, desc=f"Processing {sample_id}", leave=False)):
        frame_idx = info['frame_index']
        ts = info['timestamp']
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Save temp for embedding generation
        cv2.imwrite(temp_img_path, frame)
        
        current_time_sec = (ts - min_ts) / 1_000_000.0
        plot_timestamps.append(current_time_sec)
        
        frame_scores = {}

        # 1. Retrieve Existing Scores
        for name, scores in existing_results.items():
            if name in strategies and name not in strategies_to_compute:
                # Use existing value
                val = scores[i] if i < len(scores) else 0.0
                results[name].append(val)
                frame_scores[name] = val

        # 2. Compute New
        for name, strategy in strategies_to_compute.items():
            try:
                with suppress_output():
                    embedding = strategy.generate_embedding(temp_img_path)
                
                sim_score = 1.0
                if prev_embeddings[name] is not None:
                     sim = cosine_similarity(embedding.reshape(1, -1), prev_embeddings[name].reshape(1, -1))[0][0]
                     sim_score = max(0.0, min(1.0, sim))
                
                prev_embeddings[name] = embedding
                results[name].append(sim_score)
                frame_scores[name] = sim_score
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name].append(0.0)
                frame_scores[name] = 0.0

        # Annotate Frame with Scores
        frame_vis = frame.copy()
        
        # Order keys for display
        # strategies.keys() preserves insertion order usually
        display_items = []
        for name in strategies.keys():
            if name in frame_scores:
                display_items.append((name, frame_scores[name]))
        
        box_height = 40 + (30 * len(display_items))
        cv2.rectangle(frame_vis, (0, 0), (450, box_height), (0, 0, 0), -1)
        
        # Timestamp
        cv2.putText(frame_vis, f"Time: {current_time_sec:.2f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Strategy Scores
        y_pos = 65
        for name, score in display_items:
            status_text, color = get_status(score)
            display_name = (name[:15] + '..') if len(name) > 17 else name
            text = f"{display_name}: {score:.2f} [{status_text}]"
            cv2.putText(frame_vis, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30
            
        out.write(frame_vis)
        
    cap.release()
    out.release()
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
        
    # Re-encode video
    try:
        temp_vid = out_video_path + ".temp.mp4"
        if os.path.exists(out_video_path):
            os.rename(out_video_path, temp_vid)
            # -y overwrite, -v error quiet, libx264 for H.264, preset fast
            cmd = f"ffmpeg -y -v error -i {temp_vid} -c:v libx264 -pix_fmt yuv420p -preset fast {out_video_path}"
            os.system(cmd)
            if os.path.exists(temp_vid):
                os.remove(temp_vid)
    except Exception as e:
        print(f"FFmpeg encoding failed: {e}")

    # --- Plot ---
    plt.figure(figsize=(14, 7))
    styles = ['-', '--', '-.', ':']
    
    plot_idx = 0
    for name in strategies.keys():
        if name in results:
            sims = results[name]
            style = styles[plot_idx % len(styles)]
            plt.plot(plot_timestamps, sims, linestyle=style, alpha=0.8, linewidth=1.5, label=name)
            plot_idx += 1
        
    plt.ylim(0, 1.1)
    
    title_suffix = "(Outlier)" if is_outlier else "(Normal Sample)"
    plt.title(f"Comparison of Embedding Comparison Strategies {title_suffix}\nSample: {sample_id}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cosine Similarity (Frame-to-Frame)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_filename = f"{sample_id}_comparison_plot.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    # --- Save Results to JSON for Viewer App ---
    # Convert numpy types to native for JSON serialization
    json_results = {
        "sample_id": sample_id,
        "is_outlier": is_outlier,
        "timestamps": plot_timestamps,
        "strategies": results
    }
    json_path = os.path.join(OUTPUT_DIR, f"{sample_id}_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, cls=NumpyEncoder)
    print(f"Saved results to {json_path}")
    print(f"Saved results to {json_path}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="Force regeneration of all results")
    args = parser.parse_args()

    gt_ids = load_gt_ids(GT_FILE)
    if not gt_ids:
        print("No GT IDs found.")
        gt_ids = set()
    
    print(f"Found {len(gt_ids)} outlier samples.")

    # 2. Find Non-Outliers
    # Glob for all video files in data dir to find all available samples
    all_videos = glob.glob(os.path.join(DATA_DIR, "*.mp4"))
    all_ids = set()
    for v in all_videos:
        # Filename format: {uuid}.camera_front...
        basename = os.path.basename(v)
        if '.' in basename:
            sid = basename.split('.')[0]
            # Verify UUID length approx 36
            if len(sid) == 36:
                all_ids.add(sid)
    
    non_outlier_ids = list(all_ids - gt_ids)
    # Sort for determinism
    non_outlier_ids.sort()
    
    # Pick top 3 non-outliers
    selected_non_outliers = non_outlier_ids[:3]
    print(f"Selected {len(selected_non_outliers)} non-outlier samples for comparison.")
    
    # Combine lists: outliers first, then normal
    sorted_outliers = sorted(list(gt_ids))
    processing_list = []
    
    for sid in sorted_outliers:
        processing_list.append((sid, True)) # ID, is_outlier
        
    for sid in selected_non_outliers:
        processing_list.append((sid, False))

    # Load Models
    print("Loading Strategies...")
    strategies = {}
    
    # 1. Naive
    print("  Loading NaiveStrategy...")
    s = NaiveStrategy()
    s.load_model()
    strategies['Naive'] = s # Short name for video
    
    # 2. Foreground
    print("  Loading ForegroundStrictStrategy...")
    s = ForegroundStrictStrategy()
    s.load_model()
    strategies['Foreground'] = s
    
    # 3. TextDesc (YOLO)
    print("  Loading TextDescriptionStrategy...")
    s = TextDescriptionStrategy()
    s.load_model()
    strategies['YOLO Text'] = s
    
    # 4. VLM Caption (BLIP)
    print("  Loading VLMCaptionStrategy...")
    s = VLMCaptionStrategy()
    s.load_model()
    strategies['BLIP Caption'] = s
    
    # 5. OpenRouter Hazard (Replaces FastVLM)
    print("  Loading OpenRouterHazardStrategy...")
    try:
        s = OpenRouterHazardStrategy()
        if not os.getenv("OPENROUTER_API_KEY"):
            print("    [WARNING] OPENROUTER_API_KEY not found. Skipping OpenRouter Hazard. Set OPENROUTER_API_KEY to enable.")
        else:
            s.load_model()
            strategies['Hazard'] = s
    except Exception as e:
        print(f"    [Error] Could not load OpenRouter Strategy: {e}")
    
    # 6. FastViT Attention
    print("  Loading FastViTAttentionStrategy...")
    try:
         s = FastViTAttentionStrategy()
         s.load_model()
         strategies['ViT Attention'] = s
    except Exception as e:
         print(f"    [Error] Could not load FastViT Strategy: {e}")
    
    # Process ALL samples
    for i, (sample_id, is_outlier) in enumerate(processing_list):
        print(f"\nProcessing {i+1}/{len(processing_list)}: {sample_id} [Outlier={is_outlier}]")
        analyze_sample(sample_id, strategies, interval_seconds=0.5, regenerate=args.regenerate, is_outlier=is_outlier)

if __name__ == "__main__":
    main()
