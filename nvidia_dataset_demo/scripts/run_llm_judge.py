import os
import sys
import argparse
import glob
import json
import time
import requests
import cv2
import base64
from tqdm import tqdm

# Add project root to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.asil_metrics import parse_llm_response

# Configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.0-flash-001" # Solid vision model, often free/cheap

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_frames(video_path, num_frames=3):
    """
    Extracts evenly spaced frames from a video.
    Returns list of base64 encoded strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
        
    frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
    encoded_frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize for efficiency/cost if needed, but 224-ish is good. 
            # Let's keep original resolution or resize if huge. 
            # NuScenes is 1600x900 usually. Maybe resize to width 640?
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))
            
    cap.release()
    return encoded_frames

def call_llm(frames, api_key, model):
    """
    Calls OpenRouter API with frames.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nvidia_dataset_demo", # Required by OpenRouter
        "X-Title": "Nvidia Dataset Demo"
    }
    
    # Construct Content
    content = [
        {
            "type": "text",
            "text": """You are an Automotive Safety Expert specializing in ISO 26262.
Analyze the provided video frames from a driving scenario.
Determine the ASIL (Automotive Safety Integrity Level) risk classification by evaluating:
1. Severity (S0-S3): Potential harm to driver/passengers/pedestrians.
   - S0: No injuries
   - S1: Light/moderate injuries
   - S2: Severe/life-threatening injuries (survival probable)
   - S3: Life-threatening/fatal injuries (survival uncertain)
2. Exposure (E0-E4): Frequency/duration of such a scenario.
   - E1: Very low probability
   - E2: Low probability
   - E3: Medium probability
   - E4: High probability
3. Controllability (C0-C3): Ability of driver to avoid harm.
   - C0: Controllable in general
   - C1: Simply controllable (>99% of drivers)
   - C2: Normally controllable (>90% of drivers)
   - C3: Difficult to control or uncontrollable (<90% of drivers)

Provide your output strictly in JSON format:
{
  "severity": <int 0-3>,
  "exposure": <int 0-4>,
  "controllability": <int 0-3>,
  "reasoning": "<short explanation>"
}
"""
        }
    ]
    
    for frame_b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}"
            }
        })
        
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": content}
        ],
        "temperature": 0.1 # Low temp for deterministic outputs
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run LLM Judge for ISO 26262 scoring.")
    parser.add_argument("--api_key", type=str, help="OpenRouter API Key")
    parser.add_argument("--limit", type=int, default=10, help="Max samples to process")
    parser.add_argument("--dataset", type=str, default="nuscenes_ego", help="Dataset name")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM Model")
    parser.add_argument("--samples_dir", type=str, default=None, help="Custom samples dir")
    
    args = parser.parse_args()
    
    # Resolve API Key & Model
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    default_model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    model = args.model if args.model != DEFAULT_MODEL else default_model

    if not api_key:
        # Try loading from .env manually if not in env
        env_path = os.path.join(os.path.dirname(__file__), "../../.env")
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                    elif line.startswith("OPENROUTER_MODEL="):
                        val = line.split("=", 1)[1].strip()
                        if not os.environ.get("OPENROUTER_MODEL"):
                             default_model = val

    # Update model if it was still default and we found one in .env
    if args.model == DEFAULT_MODEL:
        model = default_model

    if not api_key:
        print("Error: API Key not found. Provide --api_key or set OPENROUTER_API_KEY.")
        return

    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.samples_dir:
        samples_dir = args.samples_dir
    else:
        # standard path: extracted_data/DATASET/samples
        # extracted_data is inside nvidia_dataset_demo
        samples_dir = os.path.join(base_dir, "extracted_data", args.dataset, "samples")
        
    output_dir = os.path.join(base_dir, "analysis_results", "ground_truth")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"asil_scores_{args.dataset}.json")
    
    # Load existing if any
    existing_data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            
    # Find videos
    pattern = os.path.join(samples_dir, "*.mp4")
    videos = glob.glob(pattern)
    videos.sort()
    
    if args.limit:
        videos = videos[:args.limit]
        
    print(f"Processing {len(videos)} videos from {samples_dir}...")
    
    results = existing_data
    processed_count = 0
    
    for video_path in tqdm(videos):
        # ID is filename without extension?
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        if video_id in results:
            continue
            
        # Extract Frames
        frames = extract_frames(video_path)
        if not frames:
            print(f"Failed to extract frames for {video_id}")
            continue
            
        # Call LLM
        response = call_llm(frames, api_key, args.model)
        
        if response and 'choices' in response:
            content = response['choices'][0]['message']['content']
            parsed = parse_llm_response(content)
            
            if parsed:
                results[video_id] = parsed
                processed_count += 1
                # Save periodically
                if processed_count % 5 == 0:
                     with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
            else:
                print(f"Failed to parse response for {video_id}: {content[:100]}...")
        else:
            print(f"LLM call failed for {video_id}")
            
        # Sleep to avoid rate limits?
        time.sleep(1)
        
    # Final Save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Finished. Saved scores to {output_file}")

if __name__ == "__main__":
    main()
