# NVIDIA Dataset Embedding Analysis Demo

This project explores various **embedding strategies** for analyzing autonomous driving data (NVIDIA text-to-driving dataset). It compares different methods of extracting semantic and visual information from video frames to find similar driving scenes.

## Features

- **10 Embedding Strategies:**
    1.  **Naive (SigLIP):** Encodes the entire image directly using SigLIP.
    2.  **Foreground Strict:** High-confidence YOLO foreground segmentation (isolates obvious objects).
    3.  **Foreground Loose:** Low-confidence segmentation with dilation (includes context).
    4.  **Video (VideoMAE):** Embeds the full video clip by processing 16 sampled frames using VideoMAE.
    5.  **Video (ViViT):** Embeds the full video clip using Video Vision Transformer (ViViT).
    6.  **Object Semantics:** "Bag of Objects" approach – detects objects (YOLO), individually captions them (BLIP), and aggregates into a detailed text description (SBERT).
    7.  **FastViT Attention:** Uses FastViT attention weights to mask the image, focusing on important regions before embedding.
    8.  **OpenRouter Description:** Uses external VLMs (via OpenRouter, e.g., `nvidia/nemotron-nano-12b-v2-vl:free`) to generate a detailed scene description.
    9.  **OpenRouter Hazard:** Uses external VLMs to focus strictly on hazards and safety-critical events.
    10. **OpenRouter Storyboard:** Samples 4 frames to create a 2x2 storyboard grid and analyzes the temporal sequence using an external VLM.

- **Browser-Based Viewer:**
    - Visualizes similarity search results.
    - Shows "Debug Inputs" (what the model actually saw: masked images, heatmaps, text).
    - Side-by-side video comparison.
  
![Viewer Screenshot](assets/viewer_screenshot_v2.png)

## Installation

```bash
pip install torch transformers sentence-transformers ultralytics flask opencv-python scikit-learn matplotlib pandas umap-learn dotenv
```

Download a subset of the PhysicalAI Autonomous Vehicles dataset:
```bash
mkdir -p nvidia_dataset_demo && wget "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/main/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip" -O nvidia_dataset_demo/dataset.zip
unzip -q nvidia_dataset_demo/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip -d
```
This should create a folder in `nvidia_dataset_demo/extracted_data`.

## Configuration

To use the **OpenRouter** strategies (and access external VLMs like GPT-4o, Claude 3.5, or NVIDIA Nemotron), you must set up your API key.

1.  Create a `.env` file in the `nvidia_dataset_demo` root directory:
    ```bash
    touch .env
    ```
2.  Add your OpenRouter API key and Model Name:
    ```bash
    OPENROUTER_API_KEY=sk-or-v1-...
    OPENROUTER_MODEL=nvidia/nemotron-nano-12b-v2-vl:free # default
    ```
    > **Note:** These strategies communicate with the [OpenRouter API](https://openrouter.ai). You can use any model available on OpenRouter by changing the `OPENROUTER_MODEL` variable.

## Usage

The project uses a pipeline script to process data, generate embeddings, and compute projections.

### 1. Run the Pipeline

The easiest way to run an analysis is using `scripts/run_pipeline.sh`:

```bash
./scripts/run_pipeline.sh --dataset nvidia_demo --strategy [STRATEGY_NAME] --limit 10
```

**Arguments:**
- `--dataset`: Name of the dataset (default: `nvidia_demo`).
- `--strategy`: The embedding strategy to use (see list below).
- `--limit`: (Optional) Limit the number of samples to process (useful for testing).

**Available Strategy Names:**
- `naive`
- `foreground_strict`
- `foreground_loose`
- `video_mae`
- `video_vit`
- `object_semantics`
- `fastvit_attention`
- `openrouter_description`
- `openrouter_hazard`
- `openrouter_storyboard`

**Manual Steps (Alternative):**
If you prefer to run steps individually:
1.  **Process Dataset:** `python3 src/processing/process_dataset.py --dataset_name nvidia_demo --limit 10`
2.  **Run Strategy:** `python3 src/analysis/run_strategy.py --strategy naive --dataset nvidia_demo`
3.  **Compute Projections:** `python3 src/analysis/compute_projections.py --dataset nvidia_demo --strategy naive`

***

### 2. Outlier Analysis (New)

Analyze outlier scores using multiple strategies and methods (IMLP, LOF, Isolation Forest, etc.).

```bash
# Run analysis for all methods
./scripts/run_new_analysis.sh

# Or specific strategy
python3 src/analysis/analyze_outliers.py --strategy video_mae
```

This generates:
- Distribution plots in `analysis_results/outlier_analysis/<strategy>/`
- Score JSONs in `analysis_results/outliers/`

***

### 3. Launch the Viewer

Start the Flask app to view results in your browser:

```bash
python3 src/viewer/app.py
```

Open **http://localhost:8080** in your browser.

## Viewer Interface

The browser-based viewer (`src/viewer/app.py`) provides a rich interface for interacting with the analysis results.

### Key Features:
1.  **Strategy Selection:**  The dropdown menu allows you to switch between different `results_*.json` files instantly.
2.  **Statistics Bar:** Displays the highest/lowest similarity scores and total pairs analyzed.
3.  **Similarity Ranking:**
    -   **Top 5 Most Similar:** Shows pairs with high cosine similarity.
    -   **Top 5 Least Similar:** Shows distinct pairs.
    -   **Leaderboard:** A sortable table of all pairs.
    -   **Embedding Space:** Interactive scatter plot (t-SNE/UMAP/PCA).
4.  **Debug Inputs:**
    -   Clicking on a pair opens a **Detail Modal**.
    -   **Debug Image/Text:** Shows exactly what the model "saw" (e.g., masked foreground, VLM caption, or storyboard grid).

## Directory Structure

- `src/`: Core source code.
    - `analysis/`: Analysis scripts (`run_strategy.py`, `compute_projections.py`).
    - `embeddings/`: Strategy implementations (`strategies.py`).
    - `processing/`: Data processing and loaders (`process_dataset.py`, `loaders.py`).
    - `viewer/`: Flask application (`app.py`) and templates.
    - `training/`: Training scripts (`train.py`).
    - `models/`: Model definitions and factories.
- `scripts/`: Helper scripts.
    - `run_pipeline.sh`: Main entry point.
    - `run_strategy_pipeline.sh`: Strategy orchestration.
    - `train_baseline.py`: Baseline training script.
- `analysis_results/`: Generated JSON results and debug images.
