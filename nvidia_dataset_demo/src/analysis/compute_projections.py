import argparse
import os
import glob
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from abc import ABC, abstractmethod

OUTLIER_FILE_MAP = {
    "nvidia_demo": "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nvidia_demo/calibration/worst-ade-log-10-90pctl-filtered.json",
    "nuscenes_ego": "/home/jadelynn/cf-reasoning/nvidia_dataset_demo/extracted_data/nuscenes_ego/calibration/nuscenes_percentile_outliers.json"
}

# Optional UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not found, skipping UMAP reduction.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(SRC_DIR, "../extracted_data")
OUTPUT_ROOT = os.path.join(SRC_DIR, "../analysis_results")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DatasetProjector(ABC):
    def __init__(self, dataset_name, strategy_name, outlier_file):
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.outlier_file = outlier_file

    @abstractmethod
    def load_outliers(self, all_ids):
        """
        Returns a dict mapping sample_id -> { "category": boolean }
        e.g. { "id1": {"p90": True, "p99": False} }
        """
        pass

    def load_embeddings(self):
        emb_file = os.path.join(OUTPUT_ROOT, "embeddings", self.strategy_name, f"{self.dataset_name}.pkl")
        if not os.path.exists(emb_file):
            print(f"Error: Embeddings file not found: {emb_file}")
            return None, None
            
        with open(emb_file, "rb") as f:
            data = pickle.load(f)
        return data["ids"], data["embeddings"]

    def compute_projections(self, embeddings, random_state=42):
        n_samples = len(embeddings)
        projections = {}

        # PCA
        if n_samples >= 2:
            pca = PCA(n_components=2)
            projections['pca'] = pca.fit_transform(embeddings)
        else:
            projections['pca'] = np.zeros((n_samples, 2))

        # t-SNE
        if n_samples > 1:
            perp = min(30, n_samples - 1)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, init='random', learning_rate='auto')
            projections['tsne'] = tsne.fit_transform(embeddings)
        else:
            projections['tsne'] = np.zeros((n_samples, 2))

        # UMAP
        if HAS_UMAP and n_samples > 2:
            try:
                n_neigh = min(15, n_samples - 1)
                umap_model = umap.UMAP(n_components=2, n_neighbors=n_neigh, random_state=random_state)
                projections['umap'] = umap_model.fit_transform(embeddings)
            except Exception as e:
                print(f"UMAP failed: {e}")
                projections['umap'] = np.zeros((n_samples, 2))
        else:
            projections['umap'] = np.zeros((n_samples, 2))

        return projections

    def compute_clusters(self, embeddings, k_values=[3, 5, 8], random_state=42):
        n_samples = len(embeddings)
        clusters = {}
        for k in k_values:
            if n_samples >= k:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                clusters[f"cluster_k{k}"] = kmeans.fit_predict(embeddings)
            else:
                clusters[f"cluster_k{k}"] = np.zeros(n_samples, dtype=int)
        return clusters

    def detect_outliers(self, embeddings, random_state=42):
        iso = IsolationForest(contamination='auto', random_state=random_state)
        preds = iso.fit_predict(embeddings)
        return (preds == -1) # True if outlier

    def process_independent(self, dataset_list, force=False):
        for dname in dataset_list:
            print(f"\n--- Processing Independent: {dname} ---")
            
            PROJ_DIR = os.path.join(OUTPUT_ROOT, "projections", self.strategy_name)
            out_json = os.path.join(PROJ_DIR, f"{dname}.json")
            
            if os.path.exists(out_json) and not force:
                print(f"Projections already exist at {out_json}. Skipping.")
                continue

            # 1. Load Embeddings for single dataset
            emb_file = os.path.join(OUTPUT_ROOT, "embeddings", self.strategy_name, f"{dname}.pkl")
            if not os.path.exists(emb_file):
                print(f"Skipping {dname}, no embeddings found.")
                continue
                
            with open(emb_file, "rb") as f:
                data = pickle.load(f)
            
            ids = data["ids"]
            embeddings = data["embeddings"]
            
            if len(ids) == 0:
                print(f"No embeddings in {dname}")
                continue
                
            # 2. Compute Projections & Attributes
            projections = self.compute_projections(embeddings)
            cluster_map = self.compute_clusters(embeddings)
            is_outlier_calc = self.detect_outliers(embeddings)
            
            # 3. Load Logical Outliers
            # Factory for specific logic
            outlier_file = OUTLIER_FILE_MAP.get(dname)
            if dname == "nvidia_demo":
                loader = NvidiaProjector(dname, self.strategy_name, outlier_file)
            elif dname == "nuscenes_ego":
                loader = NuScenesProjector(dname, self.strategy_name, outlier_file)
            else:
                loader = NvidiaProjector(dname, self.strategy_name, outlier_file)
            
            gt_outlier_map = loader.load_outliers(ids)
            all_outlier_keys = set()
            for _, flags in gt_outlier_map.items():
                all_outlier_keys.update(flags.keys())
                
            # 4. Construct Output
            points = []
            for i in range(len(ids)):
                pid = ids[i]
                
                # Logical Outliers
                outliers_obj = {k: False for k in all_outlier_keys}
                if pid in gt_outlier_map:
                    outliers_obj.update(gt_outlier_map[pid])
                    
                point_data = {
                    "id": pid,
                    "gt_outliers": outliers_obj,
                    "projections": {}
                }
                
                common_is_outlier = bool(is_outlier_calc[i])
                
                for method in projections:
                    coords = projections[method][i]
                    point_data["projections"][method] = {
                        "x": coords[0],
                        "y": coords[1],
                        "is_outlier": common_is_outlier
                    }
                    
                    for k_key, k_vals in cluster_map.items():
                        point_data["projections"][method][k_key] = int(k_vals[i])
                        
                points.append(point_data)
                
            # 5. Save
            os.makedirs(PROJ_DIR, exist_ok=True)
            
            with open(out_json, "w") as f:
                json.dump({"points": points}, f, indent=2, cls=NumpyEncoder)
            print(f"Saved INDEPENDENT projections for {dname} to {out_json}")

    def process_global(self, dataset_list, force=False):
        print(f"\n--- Processing Global Combined: {dataset_list} ---")
        
        PROJ_DIR = os.path.join(OUTPUT_ROOT, "projections", self.strategy_name)
        out_json = os.path.join(PROJ_DIR, "global.json")
        
        if os.path.exists(out_json) and not force:
            print(f"Global projections already exist at {out_json}. Skipping.")
            return

        # 1. Load All Embeddings
        all_ids = []
        all_embeddings = []
        all_datasets_source = [] # Track which dataset each point came from
        dataset_ranges = {} # dataset_name -> (start_idx, end_idx)
        current_idx = 0
        
        for dname in dataset_list:
            emb_file = os.path.join(OUTPUT_ROOT, "embeddings", self.strategy_name, f"{dname}.pkl")
            if not os.path.exists(emb_file):
                print(f"Skipping {dname}, no embeddings found.")
                continue
                
            with open(emb_file, "rb") as f:
                data = pickle.load(f)
                
            ids = data["ids"]
            emb = data["embeddings"]
            
            count = len(ids)
            all_ids.extend(ids)
            all_datasets_source.extend([dname] * count)
            
            if len(all_embeddings) == 0:
                all_embeddings = emb
            else:
                all_embeddings = np.concatenate((all_embeddings, emb), axis=0)
                
            dataset_ranges[dname] = (current_idx, current_idx + count)
            current_idx += count
            
        if len(all_ids) == 0:
            print("No embeddings found for any dataset.")
            return

        print(f"Joint Projection: {len(all_ids)} samples.")
        
        # 2. Compute Joint Projections
        projections = self.compute_projections(all_embeddings)
        cluster_map = self.compute_clusters(all_embeddings)
        is_outlier_calc = self.detect_outliers(all_embeddings)

        # 3. Load Outliers (Per dataset logic, but applied to global list)
        # We need to load outliers for each dataset and map them to the global IDs
        # To avoid re-loading files many times or complex logic, we can just iterate datasets again
        # OR just load them all into a big map first?
        # Let's use the 'dataset_ranges' to delegate to the specific loaders.
        
        global_gt_outlier_map = {} # id -> map
        all_outlier_keys = set()
        
        for dname, (start, end) in dataset_ranges.items():
            subset_ids = all_ids[start:end]
            outlier_file = OUTLIER_FILE_MAP.get(dname)
            
            if dname == "nvidia_demo":
                loader = NvidiaProjector(dname, self.strategy_name, outlier_file)
            elif dname == "nuscenes_ego":
                loader = NuScenesProjector(dname, self.strategy_name, outlier_file)
            else:
                loader = NvidiaProjector(dname, self.strategy_name, outlier_file)
                
            ds_map = loader.load_outliers(subset_ids)
            global_gt_outlier_map.update(ds_map)
            
            for _, flags in ds_map.items():
                all_outlier_keys.update(flags.keys())

        # 4. Construct Output (ALL points in one file)
        points = []
        for i in range(len(all_ids)):
            pid = all_ids[i]
            dname = all_datasets_source[i]
            
            # Logical Outliers
            outliers_obj = {k: False for k in all_outlier_keys}
            if pid in global_gt_outlier_map:
                outliers_obj.update(global_gt_outlier_map[pid])
                
            point_data = {
                "id": pid,
                "dataset": dname, # Crucial for global file
                "gt_outliers": outliers_obj,
                "projections": {}
            }
            
            common_is_outlier = bool(is_outlier_calc[i])
            
            for method in projections:
                coords = projections[method][i]
                point_data["projections"][method] = {
                    "x": coords[0],
                    "y": coords[1],
                    "is_outlier": common_is_outlier
                }
                
                for k_key, k_vals in cluster_map.items():
                    point_data["projections"][method][k_key] = int(k_vals[i])
            
            points.append(point_data)
            
        # 5. Save Global File
        os.makedirs(PROJ_DIR, exist_ok=True)
        
        with open(out_json, "w") as f:
            json.dump({"points": points}, f, indent=2, cls=NumpyEncoder)
        print(f"Saved GLOBAL projections to {out_json}")


class NvidiaProjector(DatasetProjector):
    def load_outliers(self, all_ids):
        outlier_map = {} # id -> {key: bool}
        
        if not self.outlier_file or not os.path.exists(self.outlier_file):
            # print(f"Warning: Outlier file not found: {self.outlier_file}")
            return outlier_map

        print(f"Loading Nvidia outliers from {self.outlier_file}...")
        with open(self.outlier_file, 'r') as f:
            data = json.load(f)

        # Build lookup table: scene_id -> list of outlier timestamps
        outlier_lookup = {}
        if "results" in data:
            results = data["results"]
            for key, val in results.items():
                if "top3_worst" in val:
                    scene_id = val.get("scene_id")
                    if scene_id not in outlier_lookup:
                        outlier_lookup[scene_id] = []
                    
                    for item in val["top3_worst"]:
                        if "t_rel_us" in item:
                            ts = int(item["t_rel_us"])
                            outlier_lookup[scene_id].append(ts)
        else:
             print("Warning: 'results' key not found in Nvidia outlier file.")
             return outlier_map

        # Match logic: Approximate match (nearest within tolerance)
        TOLERANCE_US = 1500000 / 2  # 0.75s (half of stride)
        
        match_count = 0
        for pid in all_ids:
            # pid format: {UUID}_{timestamp_us}
            try:
                parts = pid.split('_')
                ts_sample = int(parts[-1])
                uuid_sample = "_".join(parts[:-1])
            except ValueError:
                continue
                
            if uuid_sample in outlier_lookup:
                # Check for near match
                for ts_outlier in outlier_lookup[uuid_sample]:
                    diff = abs(ts_sample - ts_outlier)
                    if diff <= TOLERANCE_US:
                        if pid not in outlier_map:
                            outlier_map[pid] = {}
                        outlier_map[pid]["ade_p90"] = True
                        match_count += 1
                        break # One match enough per sample
        
        print(f"Matched {match_count} outliers out of {len(all_ids)} samples.")
        return outlier_map

class NuScenesProjector(DatasetProjector):
    def load_outliers(self, all_ids):
        outlier_map = {}
        
        if not self.outlier_file or not os.path.exists(self.outlier_file):
            return outlier_map

        print(f"Loading NuScenes outliers from {self.outlier_file}...")
        with open(self.outlier_file, 'r') as f:
             data = json.load(f)

        match_count = 0
        if "generated_samples" in data:
            samples = data["generated_samples"]
            
            for pid in all_ids:
                if pid in samples:
                    info = samples[pid]
                    if pid not in outlier_map:
                        outlier_map[pid] = {}
                    
                    if info.get("is_p90"):
                        outlier_map[pid]["surprise_potential_p90"] = True
                    if info.get("is_p99"):
                        outlier_map[pid]["surprise_potential_p99"] = True
                    
                    if outlier_map[pid]:
                        match_count += 1
        else:
            print("Warning: 'generated_samples' key not found in NuScenes outlier file.")
            
        print(f"Matched {match_count} outliers out of {len(all_ids)} samples.")
        return outlier_map


def main():
    parser = argparse.ArgumentParser(description="Compute Projections: Independent & Global")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name")
    parser.add_argument("--datasets", type=str, default=None, help="Comma separated list of datasets. If empty, scans dir.")
    parser.add_argument("--force", action="store_true", help="Force re-compute projections even if files exist")
    args = parser.parse_args()

    # Find datasets
    if args.datasets:
        target_datasets = args.datasets.split(",")
    else:
        # Scan embeddings dir
        emb_dir = os.path.join(OUTPUT_ROOT, "embeddings", args.strategy)
        if not os.path.exists(emb_dir):
            print(f"No embeddings found for {args.strategy}")
            return
        files = glob.glob(os.path.join(emb_dir, "*.pkl"))
        target_datasets = [os.path.splitext(os.path.basename(f))[0] for f in files]
        
    if not target_datasets:
        print("No datasets to process.")
        return

    print(f"Found datasets: {target_datasets}")

    # Driver instance
    driver = NvidiaProjector("dummy", args.strategy, None)
    
    # 1. Process Independent
    driver.process_independent(target_datasets, force=args.force)
    
    # 2. Process Global
    driver.process_global(target_datasets, force=args.force)

if __name__ == "__main__":
    main()

