# NVIDIA Dataset Embedding Analysis Demo

This project explores various **embedding strategies** for analyzing autonomous driving data (NVIDIA text-to-driving and NuScenes datasets). It leverages state-of-the-art computer vision models to extract semantic, visual, and spatial information to identify "difficult" or outlier driving scenes.

## Features

### 🚀 Embedding Strategies
The system supports a wide range of strategies grouped by methodology:

#### 1. Visual & Attention-Based
- **Naive (SigLIP)**: Encodes the entire image directly using SigLIP.
- **Foreground Strict**: High-confidence YOLO segmentation masks to isolate obvious objects.
- **Foreground Loose**: Low-confidence segmentation with dilation to include more object context.
- **FastViT Attention**: Uses FastViT attention weights to mask the image, focusing on salient regions.

#### 2. Temporal & Video-Centric
- **InternVideo**: Advanced video representation using InternVideo-6B for temporal feature extraction.
- **VideoMAE**: Uses masked autoencoder pre-training for video clip embedding.
- **ViViT**: Video Vision Transformer for processing sampled frame sequences.

(this is somewhat of a work in progress so may not work)

#### 3. Semantic & Object-Centric
- **Object Semantics**: Detecting objects (YOLO), individually captioning them (BLIP), and aggregating into an SBERT "Bag of Objects" description.
- **Object Counts**: Uses raw counts of detected objects (e.g., "3 cars, 1 pedestrian") as a feature vector.
- **Semantic Counts**: Distribution-based representation of object classes in the scene.

#### 4. Spatial Graph (Advanced)
- **Object Graph**: **[New]** Represents the scene as a spatial graph.
    - **Nodes**: Ego-vehicle and objects (cars, trucks, people).
    - **Visual Features**: Contextualized node embeddings via DINOv2 Spatial Token Pooling.
    - **Spatial Logic**: 3D Depth estimation (DepthAnythingV2) and relative spatial relations (Distance, Bearing, Depth-Difference).
    - **Output**: Generates a pooled embedding for pipelines and a detailed `.json` graph for GNN training.

#### 5. VLM-Powered (OpenRouter)
- **OpenRouter Description**: Full scene analysis via VLMs (e.g., GPT-4o, NVIDIA Nemotron).
- **OpenRouter Hazard**: Focused hazard and risk assessment.
- **OpenRouter Storyboard**: Temporal sequence analysis of sampled frames in a grid.

---

## Outlier Analysis Framework

The project includes an anomaly detection suite to identify rare or difficult scenarios:
- **Algorithms**: Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, and IMLP.
- **Metrics**: AUROC and AUPRC against ground-truth interactivity/ASIL scores.
- **Visualizations**: Comparative distribution plots and embedding space projections (UMAP/t-SNE).

---

## Installation

```bash
pip install torch transformers sentence-transformers ultralytics flask opencv-python scikit-learn matplotlib pandas umap-learn python-dotenv
```

## Usage

### 1. Run Analysis Pipeline
Use the provided script to process data and generate embeddings:

```bash
# Example: Run Object Graph Strategy on NuScenes
python3 src/analysis/run_strategy.py --strategy object_graph --dataset nuscenes_ego
```

### 2. Compute Projections & Identify Outliers
```bash
# Compute 2D projections (UMAP/t-SNE)
python3 src/analysis/compute_projections.py --dataset nuscenes_ego --strategy object_graph

# Run the outlier detection suite
python3 src/analysis/analyze_outliers.py --strategy object_graph
```

### 3. Launch the Viewer
Start the Flask app to visualize results:
```bash
python3 src/viewer/app.py
```
Open **http://localhost:8080** to explore the embedding space, similarity matches, and graph visualizations.

---

## Directory Structure
- `src/`:
    - `analysis/`: Outlier detection, projections, and strategy runners.
    - `embeddings/`: Strategy implementations (`strategies.py`) and specialized models.
    - `processing/`: Data loaders and dataset processing.
    - `viewer/`: Flask application and interactive UI.
- `analysis_results/`: Generated embeddings, outlier scores, and debug visualizations.
- `extracted_data/`: Dataset storage (NVIDIA demo, NuScenes).
- `scripts/`: Pipeline orchestration and helper scripts.
