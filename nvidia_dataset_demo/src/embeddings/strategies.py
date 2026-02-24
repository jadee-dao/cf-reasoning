import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import numpy as np
import cv2
from .base import EmbeddingStrategy
from .prompts import SCENE_DESCRIPTION_PROMPT, HAZARD_IDENTIFICATION_PROMPT

import os 

# --- Constants ---
SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
SBERT_MODEL_NAME = "all-mpnet-base-v2"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
# Point to models directory
YOLO_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/yolo11n.pt"))
DINOV2_MODEL_NAME = "facebook/dinov2-small"
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ... (Naive, Foreground, TextDescription classes remain unchanged) ...

class VideoMAEStrategy(EmbeddingStrategy):
    """
    Uses VideoMAE (videomae-base) to embed video clips.
    Samples 16 frames and computes the mean of the encoder outputs.
    """
    def load_model(self):
        if not self.model_loaded:
            model_name = "MCG-NJU/videomae-base"
            print(f"Loading VideoMAE model: {model_name}...")
            
            from transformers import VideoMAEImageProcessor, VideoMAEModel
            
            self.device = get_device()
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.model = VideoMAEModel.from_pretrained(model_name).to(self.device)
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        video_path = kwargs.get("video_path")
        if not video_path:
            raise ValueError("Video path not provided for VideoMAE Strategy")

        # 1. Sample 16 Frames (VideoMAE default)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Uniform sampling of 16 frames
        indices = np.linspace(0, total_frames-1, 16).astype(int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        
        # 2. Prepare Debug Filmstrip
        if "debug_output_path" in kwargs:
            target_h = 100
            scale = target_h / frames[0].shape[0]
            target_w = int(frames[0].shape[1] * scale)
            
            # Show every other frame for compactness (8 frames)
            resized_frames = [cv2.resize(f, (target_w, target_h)) for f in frames[::2]]
            filmstrip = np.concatenate(resized_frames, axis=1)
            Image.fromarray(filmstrip).save(kwargs["debug_output_path"])

        # 3. Generate Embedding
        # VideoMAE expects list of frames
        inputs = self.processor(list(frames), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use mean of last hidden state as embedding
        # [Batch, Sequence, Dim] -> [1, 1568, 768] for base model
        last_hidden_state = outputs.last_hidden_state
        embedding = torch.mean(last_hidden_state, dim=1)
            
        # Normalize
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()



class NaiveStrategy(EmbeddingStrategy):
    """
    Computes embedding of the full image using SigLIP.
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SigLIP model: {SIGLIP_MODEL_NAME}...")
            # Use device_map="auto" to handle OOM gracefully (requires accelerate)
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, device_map="auto")
            self.model_loaded = True
            
    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        image = Image.open(image_path).convert("RGB")
        
        if "debug_output_path" in kwargs:
            image.save(kwargs["debug_output_path"])
            
        # Move inputs to the same device as the model
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()


class ForegroundStrictStrategy(EmbeddingStrategy):
    """
    Strict Foreground: Now uses the previous 'Loose' settings.
    Confidence 0.1 + Moderate Dilation. Captures objects and immediate context.
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SigLIP model: {SIGLIP_MODEL_NAME}...")
            self.device = get_device()
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device)
            
            print(f"Loading YOLO Segmentation model: {YOLO_MODEL_PATH}...")
            self.seg_model = YOLO("yolo11n-seg.pt") 
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        # Strict (formerly Loose): conf=0.1
        results = self.seg_model(image_path, verbose=False, conf=0.1)
        result = results[0]
        
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            for mask in result.masks.data:
                m = mask.cpu().numpy()
                m = cv2.resize(m, (w, h))
                combined_mask = np.maximum(combined_mask, m)
        
        combined_mask = (combined_mask > 0.5).astype(np.uint8)
        
        # Moderate Dilation
        kernel = np.ones((15, 15), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        foreground = cv2.bitwise_and(img, img, mask=combined_mask)
        foreground_pil = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
        
        if "debug_output_path" in kwargs:
            foreground_pil.save(kwargs["debug_output_path"])
        
        inputs = self.processor(images=foreground_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()


class ForegroundLooseStrategy(EmbeddingStrategy):
    """
    Loose Foreground: Very Low confidence (0.01) + Aggressive Dilation.
    Includes almost everything that might be an object.
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SigLIP model: {SIGLIP_MODEL_NAME}...")
            self.device = get_device()
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device)
            
            print(f"Loading YOLO Segmentation model: {YOLO_MODEL_PATH}...")
            self.seg_model = YOLO("yolo11n-seg.pt") 
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        # Ultra Loose: conf=0.01
        results = self.seg_model(image_path, verbose=False, conf=0.01)
        result = results[0]
        
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            for mask in result.masks.data:
                m = mask.cpu().numpy()
                m = cv2.resize(m, (w, h))
                combined_mask = np.maximum(combined_mask, m)
        
        combined_mask = (combined_mask > 0.5).astype(np.uint8)
        
        # Aggressive Dilation (capture broad context)
        kernel = np.ones((15, 15), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=6) # Increased from 2 to 6
        
        foreground = cv2.bitwise_and(img, img, mask=combined_mask)
        foreground_pil = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
        
        if "debug_output_path" in kwargs:
            foreground_pil.save(kwargs["debug_output_path"])
        
        inputs = self.processor(images=foreground_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()





class ObjectSemanticsStrategy(EmbeddingStrategy):
    """
    Detects objects (YOLO), crops them, captions each individually (BLIP),
    and embeds the aggregated scene description (SBERT).
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
            
            print(f"Loading BLIP model: {BLIP_MODEL_NAME}...")
            self.device = get_device()
            self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(self.device)
            
            print(f"Loading YOLO Detection model: {YOLO_MODEL_PATH}...")
            self.det_model = YOLO("yolo11n.pt")
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        image = Image.open(image_path).convert("RGB")
        
        # 1. Run YOLO Detection
        results = self.det_model(image_path, verbose=False, conf=0.25)
        result = results[0]
        
        object_crops = []
        object_labels = [] 
        
        if result.boxes is not None:
            for box in result.boxes:
                # box.xyxy: [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                # Crop
                # Ensure coordinates are within bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image.crop((x1, y1, x2, y2))
                # Skip tiny crops
                if crop.width < 10 or crop.height < 10:
                    continue
                    
                object_crops.append(crop)
                
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                object_labels.append(label)

        captions = []
        if object_crops:
            # 2. Batch Captioning (Parallel)
            inputs = self.processor(images=object_crops, return_tensors="pt").to(self.device)
            
            # Generate
            out = self.model.generate(**inputs, max_new_tokens=20)
            captions = self.processor.batch_decode(out, skip_special_tokens=True)
            
        # 3. Global Caption (Full Image)
        global_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        global_out = self.model.generate(**global_inputs, max_new_tokens=50)
        global_caption = self.processor.decode(global_out[0], skip_special_tokens=True)

        # 4. Aggregate Description
        if captions:
            # "Scene description: a car driving down a street. Objects: a red car, a blue truck..."
            unique_captions = list(set(captions)) # Remove exact duplicates
            joined_captions = ", ".join(unique_captions)
            description = f"Scene description: {global_caption}. Objects detected: {joined_captions}."
        else:
            description = f"Scene description: {global_caption}. Objects detected: None."

        # 5. Save Debug Info
        if "debug_output_path" in kwargs:
            # Save annotated image for context
            res_plotted = result.plot()
            cv2.imwrite(kwargs["debug_output_path"], res_plotted)
            
            # Save text
            txt_path = kwargs["debug_output_path"].replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.write("\n\n--- Global Caption ---\n")
                f.write(f"{global_caption}\n")
                f.write("\n--- Individual Objects ---\n")
                if object_labels and captions:
                    for label, cap in zip(object_labels, captions):
                        f.write(f"[{label}]: {cap}\n")

        # 6. Embed
        return self.sbert.encode(description)


class FastViTAttentionStrategy(EmbeddingStrategy):
    """
    Uses FastVLM's vision encoder (FastViT) to identify important regions 
    via feature map activation, masks the image to focus on these regions, 
    and generates a 'focused' embedding.
    """
    def load_model(self):
        if not self.model_loaded:
            model_id = "apple/FastVLM-1.5B"
            print(f"Loading FastVLM model: {model_id}...")
            
            # Additional import needed for this specific model type if distinct
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.device = get_device()
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading FastVLM: {e}")
                raise e

    def _extract_vision_features(self, pil_image):
        # Helper to run vision tower
        # Check for image processor
        image_processor = None
        if hasattr(self.model.model.vision_tower, 'image_processor'):
            image_processor = self.model.model.vision_tower.image_processor
            
        if image_processor:
            inputs = image_processor(pil_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.model.device)
        else:
            # Basic fallback if needed, but FastVLM usually has it
            print("Warning: No image processor found for FastVLM.")
            return None

        with torch.no_grad():
            vision_outputs = self.model.model.vision_tower(pixel_values)
            if isinstance(vision_outputs, tuple):
                image_features = vision_outputs[0]
            else:
                image_features = vision_outputs
        return image_features

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        pil_image = Image.open(image_path).convert("RGB")
        
        # 1. First Pass: Get Features & Importance Map
        image_features = self._extract_vision_features(pil_image)
        if image_features is None:
            return np.zeros(1) # Fail safe
            
        # Determine shape [B, L, D] or [B, D, H, W]
        if len(image_features.shape) == 3: # [B, L, D]
            B, L, D = image_features.shape
            # Assuming square grid for FastViT
            H = W = int(L**0.5)
            activation_map = torch.mean(image_features, dim=2) # [B, L]
            activation_map = activation_map.reshape(B, H, W)
            importance_map = activation_map[0].cpu().numpy()
        elif len(image_features.shape) == 4: # [B, D, H, W]
            activation_map = torch.mean(image_features, dim=1) # [B, H, W]
            importance_map = activation_map[0].cpu().numpy()
        else:
            print(f"Unexpected feature shape: {image_features.shape}")
            return np.zeros(1)

        # 2. Threshold & Mask
        importance_map_norm = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())
        
        # Resize heatmap to original image
        img_np = np.array(pil_image)
        heatmap_resized = cv2.resize(importance_map_norm, (img_np.shape[1], img_np.shape[0]))
        
        # Threshold
        threshold = np.mean(heatmap_resized) + 0.5 * np.std(heatmap_resized)
        mask = heatmap_resized > threshold
        
        # Create masked image
        masked_img_np = img_np.copy()
        masked_img_np[~mask] = 0 # Black out unimportant
        masked_pil = Image.fromarray(masked_img_np)

        # 3. Save Debug Visualization
        if "debug_output_path" in kwargs:
            # Visualize: Original | Heatmap | Masked
            import matplotlib.pyplot as plt
            
            # Heatmap overlay
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(pil_image)
            ax[0].set_title("Original")
            ax[0].axis('off')
            
            ax[1].imshow(overlay)
            ax[1].set_title("Importance (FastViT)")
            ax[1].axis('off')
            
            ax[2].imshow(masked_pil)
            ax[2].set_title(f"Masked (>{threshold:.2f})")
            ax[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(kwargs["debug_output_path"])
            plt.close()

        # 4. Second Pass: Get Focused Embedding
        focused_features = self._extract_vision_features(masked_pil)
        
        # Embedding: Average Pooling
        if len(focused_features.shape) == 3:
            embedding = torch.mean(focused_features, dim=1) # [B, D]
        elif len(focused_features.shape) == 4:
            embedding = torch.mean(focused_features, dim=(2, 3)) # [B, D]
            
        # Normalize
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()





class SemanticObjectDistributionStrategy(EmbeddingStrategy):
    """
    Combines quantitative object counts (YOLO) with spatial interactivity metrics.
    Generates a description like: "High density traffic scene with 5 cars. Objects are clustered." 
    Embeds this description with SBERT.
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
            
            # BLIP no longer needed for this strategy
            
            print(f"Loading YOLO Detection model: {YOLO_MODEL_PATH}...")
            self.det_model = YOLO("yolo11n.pt")
            self.model_loaded = True

    def _analyze_interactions(self, boxes, width, height):
        """
        Analyzes bounding boxes for spatial distribution.
        boxes: List of [x1, y1, x2, y2, cls_id]
        """
        if not boxes:
            return "Empty scene", "None", []

        # 1. Pre-processing: Filter small objects and duplicates
        valid_boxes = []
        for b in boxes:
            x1, y1, x2, y2, cls_id = b
            area = (x2 - x1) * (y2 - y1)
            norm_area = area / (width * height)
            
            # Class-specific thresholds
            # Vulnerable classes (Person, Bicycle, Motorcycle) - keep even if small
            # Assuming COCO classes roughly: 0=person, 1=bicycle, 3=motorcycle... 
            # We don't have the names map here directly unless we pass it or infer.
            # But the caller passed `boxes` which we modified to include cls_id.
            # We assume standard YOLO classes: 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck
            
            is_vulnerable = cls_id in [0, 1, 3] 
            
            # Thresholds
            # Person/Bike: 0.05% (very small but visible)
            # Vehicle: 0.1% (capture distant cars that might be relevant for p90)
            threshold = 0.0005 if is_vulnerable else 0.001
            
            if norm_area > threshold: 
                valid_boxes.append(b)
                
        # Simple NMS-like deduplication (IoU check)
        unique_boxes = []
        if valid_boxes:
            # Sort by area descending
            valid_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            unique_boxes.append(valid_boxes[0])
            
            for i in range(1, len(valid_boxes)):
                b1 = valid_boxes[i]
                overlap = False
                for b2 in unique_boxes:
                    # Calculate IoU
                    xA = max(b1[0], b2[0])
                    yA = max(b1[1], b2[1])
                    xB = min(b1[2], b2[2])
                    yB = min(b1[3], b2[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    box1Area = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    box2Area = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = interArea / float(box1Area + box2Area - interArea)
                    
                    if iou > 0.8: # High overlap -> Duplicate/Same object
                        overlap = True
                        break
                if not overlap:
                    unique_boxes.append(b1)
        
        boxes = unique_boxes
        num_objs = len(boxes)
        
        if num_objs == 0:
             return "Sparse/Empty scene", "No interactions", []

        centers = []
        for b in boxes:
            x1, y1, x2, y2 = b[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx / width, cy / height))

        # 2. Density / Proximity Metrics
        if num_objs > 1:
            distances = []
            centers_np = np.array(centers)
            for i in range(num_objs):
                for j in range(i + 1, num_objs):
                    dist = np.linalg.norm(centers_np[i] - centers_np[j])
                    distances.append(dist)
            avg_dist = np.mean(distances) if distances else 0.0
            min_dist = np.min(distances) if distances else 0.0
        else:
            avg_dist = 0.0
            min_dist = 1.0
            
        # 3. Logic to text (Tuned Thresholds)
        density_desc = "Low density"
        if num_objs > 4:
            if avg_dist < 0.2: density_desc = "Very High density (crowded)"
            elif avg_dist < 0.4: density_desc = "High density"
            else: density_desc = "Moderate density"
        elif num_objs > 1:
             if avg_dist < 0.15: density_desc = "Tightly clustered group"
             else: density_desc = "Distributed objects"
             
        proximity_desc = "No close interactions"
        if num_objs > 1 and min_dist < 0.02: 
            proximity_desc = "Critical proximity detected (collision risk)"
        elif num_objs > 1 and min_dist < 0.08:
            proximity_desc = "Close interaction detected"
            
        return density_desc, proximity_desc, boxes

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        
        # 1. Run YOLO
        results = self.det_model(image_path, verbose=False, conf=0.25)
        result = results[0]
        
        raw_boxes = []
        if result.boxes is not None:
             for box in result.boxes:
                 coords = box.xyxy[0].cpu().numpy()
                 cls_id = int(box.cls[0])
                 # [x1, y1, x2, y2, cls_id]
                 b = np.append(coords, cls_id)
                 raw_boxes.append(b)

        # 2. Analyze Interactions
        density, proximity, filtered_boxes = self._analyze_interactions(raw_boxes, w, h)
        
        # Recalculate counts based on FILTERED boxes
        counts = {}
        for b in filtered_boxes:
             cls_id = int(b[4])
             label = result.names[cls_id]
             counts[label] = counts.get(label, 0) + 1
             
        if counts:
            count_parts = [f"{count} {label}{'s' if count > 1 else ''}" for label, count in counts.items()]
            count_str = ", ".join(count_parts)
        else:
            count_str = "no detected objects"
        
        # 3. Synthesize Description
        description = f"Scene Content: {count_str}. Analysis: {density}. {proximity}."
        
        # 4. Save Debug Info (Custom Drawing)
        if "debug_output_path" in kwargs:
            # Load CV2 image
            debug_img = cv2.imread(image_path)
            
            # Draw Filtered Boxes
            for b in filtered_boxes:
                x1, y1, x2, y2 = map(int, b[:4])
                cls_id = int(b[4])
                label = result.names[cls_id]
                
                # Green Box
                color = (0, 255, 0) 
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # Overlay Description Text (Top of image with background)
            banner_height = 80
            cv2.rectangle(debug_img, (0, 0), (w, banner_height), (0, 0, 0), -1)
            
            # Fit text
            cv2.putText(debug_img, f"Analysis: {density}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_img, f"{proximity}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            cv2.putText(debug_img, f"Counts: {count_str}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imwrite(kwargs["debug_output_path"], debug_img)
            
            txt_path = kwargs["debug_output_path"].replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.write(description)

        # 5. Embed
        return self.sbert.encode(description)


class ObjectCountStrategy(EmbeddingStrategy):
    """
    Generates an embedding which is simply the raw count of objects in the scene.
    Vector: [n_person, n_bicycle, n_car, n_motorcycle, n_bus, n_truck]
    
    This is a "Feature Vector" approach to test if simple object count correlates
    with interactivity outliers.
    """
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self):
        if not self.model_loaded:
             print(f"Loading YOLO Detection model: {YOLO_MODEL_PATH}...")
             try:
                 from ultralytics import YOLO
                 self.model = YOLO("yolo11n.pt") 
                 self.model_loaded = True
             except ImportError:
                 print("Error: ultralytics not installed. Please install it.")
                 self.model = None

    def generate_embedding(self, image_path, video_path=None, **kwargs):
        self.load_model()
        if self.model is None:
            return np.zeros(6, dtype=np.float32)
            
        results = self.model(image_path, verbose=False)
        result = results[0]
        
        # COCO Classes:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        target_classes = [0, 1, 2, 3, 5, 7]
        counts = {cls_id: 0 for cls_id in target_classes}
        
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in counts:
                    counts[cls_id] += 1
                
        # Create vector
        # [Person, Bike, Car, Motor, Bus, Truck]
        vector = [
            counts[0], # Person
            counts[1], # Bike
            counts[2], # Car
            counts[3], # Motorcycle
            counts[5], # Bus
            counts[7]  # Truck
        ]
        
        # Return as numpy array (float32 for compatibility)
        return np.array(vector, dtype=np.float32)

# --- OpenRouter Strategies ---

import requests
import base64
import os


class ViViTStrategy(EmbeddingStrategy):
    """
    Uses ViViT (Video Vision Transformer) to embed video clips.
    Model: google/vivit-b-16x2-kinetics400
    Native video processing without masking.
    """
    def load_model(self):
        if not self.model_loaded:
            model_name = "google/vivit-b-16x2-kinetics400"
            print(f"Loading ViViT model: {model_name}...")
            
            from transformers import VivitImageProcessor, VivitModel
            
            self.device = get_device()
            self.processor = VivitImageProcessor.from_pretrained(model_name)
            self.model = VivitModel.from_pretrained(model_name).to(self.device)
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        video_path = kwargs.get("video_path")
        if not video_path:
            raise ValueError("Video path not provided for ViViT Strategy")

        # 1. Sample Frames (ViViT expects 32 frames by default usually)
        # The specific model 'google/vivit-b-16x2-kinetics400' typically expects 32 frames.
        # We will sample 32 frames.
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
             # Fallback
             total_frames = 32
        
        # Uniform sampling of 32 frames
        indices = np.linspace(0, total_frames-1, 32).astype(int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        
        # 2. Prepare Debug Filmstrip (Optional, same as VideoMAE)
        if "debug_output_path" in kwargs:
            target_h = 100
            scale = target_h / frames[0].shape[0]
            target_w = int(frames[0].shape[1] * scale)
            
            # Show every 4th frame for compactness (8 frames)
            resized_frames = [cv2.resize(f, (target_w, target_h)) for f in frames[::4]]
            filmstrip = np.concatenate(resized_frames, axis=1)
            Image.fromarray(filmstrip).save(kwargs["debug_output_path"])

        # 3. Generate Embedding
        # VivitProcessor expects list of frames
        inputs = self.processor(list(frames), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use pooler_output (CLS token embedding)
        # Shape: [Batch, Hidden] -> [1, 768]
        embedding = outputs.pooler_output[0]
            
        # Normalize
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()


class OpenRouterDescriptionStrategy(EmbeddingStrategy):
    """
    Uses OpenRouter API to describe the scene, then embeds the description using SBERT.
    """
    def __init__(self, prompt_text=None, model=None):
        super().__init__()
        self.sbert = None
        self.model_loaded = False
        self.prompt_text = prompt_text or SCENE_DESCRIPTION_PROMPT
        # Default to the requested model
        self.model_name = model or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def get_config_name(self):
        # Sanitize model name for filename (replace / and : with _)
        sanitized = self.model_name.replace("/", "_").replace(":", "_")
        return f"{sanitized}_description"

    def load_model(self):
        if not self.model_loaded:
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")
                
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
            self.model_loaded = True

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        # 1. Prepare Image (Base64)
        # If it's a video, we might need to extract a frame first if not already done by caller.
        # run_embedding_test.py extracts a temp frame, so image_path is likely a .jpg
        if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
             cap = cv2.VideoCapture(image_path)
             ret, frame = cap.read()
             cap.release()
             if ret:
                 # Encode from memory
                 _, buffer = cv2.imencode('.jpg', frame)
                 base64_image = base64.b64encode(buffer).decode('utf-8')
             else:
                 raise ValueError("Could not read frame from video")
        else:
            base64_image = self._encode_image(image_path)

        # 2. Call OpenRouter API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vlabench", # Optional
            "X-Title": "VLABench Dataset Demo" # Optional
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        max_retries = 3
        description = ""
        
        for attempt in range(max_retries):
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                description = data['choices'][0]['message']['content'].strip()
                break
            except Exception as e:
                print(f"OpenRouter API Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Falling back to empty description.")
                    description = "analysis failed"

        print(f"\n[OpenRouter {self.model_name}]:\n{description}\n")

        # 3. Save Debug Info
        if "debug_output_path" in kwargs:
            txt_path = kwargs["debug_output_path"].replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.write(description)
            
            # Save copy of image
            if image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                import shutil
                shutil.copy(image_path, kwargs["debug_output_path"])
            else:
                # If it was video/memory, we save what we have
                pass 

        # 4. Embed with SBERT
        return self.sbert.encode(description)

class OpenRouterHazardStrategy(OpenRouterDescriptionStrategy):
    def __init__(self):
        super().__init__(prompt_text=HAZARD_IDENTIFICATION_PROMPT)

    def get_config_name(self):
        sanitized = self.model_name.replace("/", "_").replace(":", "_")
        return f"{sanitized}_hazard"

class OpenRouterStoryboardStrategy(OpenRouterDescriptionStrategy):
    """
    Samples 4 frames from the video, updates them to a 2x2 grid, 
    and prompts the VLM to analyze the sequence for difficulty and hazards.
    """
    def __init__(self):
        # We'll use the specific STORYBOARD prompt
        from .prompts import STORYBOARD_PROMPT
        super().__init__(prompt_text=STORYBOARD_PROMPT)

    def get_config_name(self):
        sanitized = self.model_name.replace("/", "_").replace(":", "_")
        return f"{sanitized}_storyboard"

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        # 1. Get Video Path
        # Ideally, run_embedding_test.py passes 'video_path'. 
        # If not, we might fail or scrape from image_path.
        video_path = kwargs.get("video_path")
        if not video_path:
            # Fallback (dangerous if file structure changes)
            # image_path is typically .../temp_frame_ID.jpg
            # video is often at data_dir/ID.mp4 or extracted_data/.../ID.mp4?
            # Let's assume run_embedding_test ALWAYS passes video_path now.
            raise ValueError("Video path required for Storyboard Strategy")

        # 2. Sample 4 Frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 4:
            # Not enough frames, just duplicate?
            indices = np.linspace(0, total_frames-1, total_frames).astype(int)
            # Pad to 4
            while len(indices) < 4:
                indices = np.append(indices, indices[-1])
        else:
            # 10%, 35%, 60%, 85%
            indices = [
                int(total_frames * 0.10),
                int(total_frames * 0.35),
                int(total_frames * 0.60),
                int(total_frames * 0.85)
            ]
            
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for grid (e.g., 384x384 each -> 768x768 total)
                # Keep it reasonable for API costs/limits
                frame = cv2.resize(frame, (384, 384))
                frames.append(frame)
            else:
                # Black frame fallback
                frames.append(np.zeros((384, 384, 3), dtype=np.uint8))
        cap.release()
        
        # 3. Stitch 2x2 Grid
        # Top Row: Frame 0, Frame 1
        top_row = np.hstack((frames[0], frames[1]))
        # Bottom Row: Frame 2, Frame 3
        bot_row = np.hstack((frames[2], frames[3]))
        # Full Grid
        grid_img = np.vstack((top_row, bot_row))
        
        # 4. Encode Grid Image
        _, buffer = cv2.imencode('.jpg', grid_img)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # 5. Call API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vlabench",
            "X-Title": "VLABench Storyboard"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Retry logic similar to parent
        max_retries = 3
        description = ""
        for attempt in range(max_retries):
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                description = data['choices'][0]['message']['content'].strip()
                break
            except Exception as e:
                print(f"OpenRouter Storyboard Error (Attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    description = "storyboard analysis failed"

        print(f"\n[OpenRouter {self.model_name} Storyboard]:\n{description}\n")

        # 6. Save Debug Info
        if "debug_output_path" in kwargs:
            txt_path = kwargs["debug_output_path"].replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.write(description)


class InternVideoStrategy(EmbeddingStrategy):
    """
    Uses InternVideo (based on VideoMAE V2) to embed video clips.
    Model: OpenGVLab/internvideo-mm-l-14 
    This implementation follows VideoMAE structure but uses the enhanced V2 model via AutoModel.
    """
    def load_model(self):
        if not self.model_loaded:
            # User requested specific model
            model_name = "OpenGVLab/InternVideo2_5_Chat_8B" 
            print(f"Loading InternVideo 2.5 Chat model: {model_name}...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.device = get_device()
            # 8B model requires device_map="auto" to fit/offload
            # trust_remote_code is REQUIRED for InternVideo models
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True, 
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                print("Make sure you are logged in: 'hf auth login'")
                raise e

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        video_path = kwargs.get("video_path")
        if not video_path:
            raise ValueError("Video path not provided for InternVideo Strategy")

        # 1. Sample Frames
        # InternVideo2.5 typically uses 8-16 frames. 
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 16
        
        # Sample 16 frames
        indices = np.linspace(0, total_frames-1, 16).astype(int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        
        # 2. Debug Filmstrip
        if "debug_output_path" in kwargs:
            target_h = 100
            scale = target_h / frames[0].shape[0]
            target_w = int(frames[0].shape[1] * scale)
            resized_frames = [cv2.resize(f, (target_w, target_h)) for f in frames[::2]]
            filmstrip = np.concatenate(resized_frames, axis=1)
            Image.fromarray(filmstrip).save(kwargs["debug_output_path"])

        # 3. Generate Embedding (Vision Encode Only)
        # InternVideo2.5 usually has a 'extract_feature' or we access vision_tower directly
        # The model class is usually InternVLChatModel
        
        # Prepare inputs: Raw frames -> Pixel Values
        # Note: The model's image processor logic is complex, often using dynamic resolution.
        # We will try the simplest path: model.encode_video usually exists or visual_encoder
        
        try:
            # Convert to tensor [T, C, H, W] or [1, T, C, H, W]
            # Standard resize to 224 for embedding speed/consistency unless model enforces higher
            pixel_values = []
            for f in frames:
                f_resized = cv2.resize(f, (224, 224))
                f_tensor = torch.from_numpy(f_resized).permute(2, 0, 1).float() / 255.0
                pixel_values.append(f_tensor)
            
            # [T, C, H, W]
            video_tensor = torch.stack(pixel_values) 
            
            # Normalize (ImageNet mean/std usually)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            video_tensor = (video_tensor - mean) / std
            
            # Add batch dim [1, T, C, H, W]
            video_tensor = video_tensor.unsqueeze(0).to(self.device).half()

            with torch.no_grad():
                # Attempt 1: visual_encoder directly
                if hasattr(self.model, 'vision_model'):
                    # InternVL structure
                    # [B*T, C, H, W] -> flatten batch/time for vision tower?
                    b, t, c, h, w = video_tensor.shape
                    # reshape to [B*T, C, H, W]
                    vision_input = video_tensor.view(-1, c, h, w)
                    vision_features = self.model.vision_model(vision_input)
                    # Output shape: [B*T, num_patches, hidden_size]
                    # Average pool over patches and time
                    # vision_features.last_hidden_state
                    feats = vision_features.last_hidden_state
                    embedding = torch.mean(feats, dim=(0, 1)) # Mean over T and Patches
                    
                elif hasattr(self.model, 'extract_feature'):
                    # Some variants have this
                    embedding = self.model.extract_feature(video_tensor)
                    
                else:
                    # Fallback: Forward pass might return full multimodal output
                    # This is tricky for Chat models without text input.
                    # We will try to find 'vision_tower'
                    vision_tower = getattr(self.model, 'vision_tower', None)
                    if vision_tower:
                         # Similar reshaping as above might be needed
                         b, t, c, h, w = video_tensor.shape
                         vision_input = video_tensor.view(-1, c, h, w)
                         feats = vision_tower(vision_input)
                         # feats might be a list or tuple, take last
                         if isinstance(feats, (list, tuple)): feats = feats[-1]
                         embedding = torch.mean(feats, dim=(0, 1))
            return embedding.cpu().numpy()
            
        except Exception as e:
            print(f"Embedding failed: {e}")
            # Identify failure type (OOM vs API)
            return np.zeros(1024) # Placeholder

class ObjectGraphStrategy(EmbeddingStrategy):
    """
    Represent scene as a spatial graph:
    - Nodes: Ego-vehicle + detected objects.
    - Node Features: DINOv2 token pooling (patch-aligned features) + Depth + Metadata.
    - Edges: Directed relative-to-ego spatial relationships (Distance, Depth, Bearing).
    - Output: A pooled graph embedding for pipeline compatibility + detailed JSON export.
    """
    def __init__(self):
        super().__init__()
        self.seg_model = None
        self.depth_model = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.depth_processor = None

    def load_model(self):
        if not self.model_loaded:
            print(f"Loading YOLOv11-seg...")
            self.seg_model = YOLO("yolo11n-seg.pt")
            
            print(f"Loading DepthAnythingV2...")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME).to(get_device())
            
            print(f"Loading DINOv2...")
            from transformers import BitImageProcessor, Dinov2Model
            self.dinov2_processor = BitImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
            self.dinov2_model = Dinov2Model.from_pretrained(DINOV2_MODEL_NAME).to(get_device())
            
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        device = get_device()
        
        # 1. Load Image
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        img_cv2 = cv2.imread(image_path)
        
        # 2. Run Segmentation (YOLOv11-seg)
        results = self.seg_model(image_path, verbose=False, conf=0.25)
        result = results[0]
        
        # 3. Run Depth Estimation
        depth_inputs = self.depth_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            depth_outputs = self.depth_model(**depth_inputs)
            # DepthAnythingV2 outputs predicted depth
            predicted_depth = depth_outputs.predicted_depth 
            # Resize to original image size
            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            
        # Normalize depth for feature use [0, 1] - DepthAnythingV2 typically outputs higher values for closer objects
        # We want 0 = closest (ego) and 1 = furthest for intuitive distance calcs
        depth_mean = depth_map.mean()
        depth_std = depth_map.std()
        # Standard min-max normalization
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        # Invert so 0 is close, 1 is far
        depth_norm = 1.0 - depth_norm

        # 4. Run DINOv2 for Spatial Tokens
        # DINOv2 expects image size to be multiple of patch size (14)
        # Standard size for small is often 518x518 (37*14)
        dinov2_size = 518
        # Ensure image is resized to a multiple of 14 for the vision transformer
        image_resized = image.resize((dinov2_size, dinov2_size))
        dinov2_inputs = self.dinov2_processor(images=image_resized, return_tensors="pt").to(device)
        with torch.no_grad():
            dinov2_outputs = self.dinov2_model(**dinov2_inputs)
            # [1, 1 + L, 384]
            last_hidden_state = dinov2_outputs.last_hidden_state
            
            # Dynamically calculate grid size
            num_tokens = last_hidden_state.shape[1] - 1
            grid_size = int(np.sqrt(num_tokens))
            if grid_size * grid_size != num_tokens:
                 print(f"Warning: tokens {num_tokens} not a perfect square.")
            
            # Exclude CLS token and reshape
            patch_tokens = last_hidden_state[:, 1:, :].reshape(grid_size, grid_size, 384)
            # Update grid size for later use
            PATCH_GRID_SIZE = grid_size
            
        # 5. Build Graph Nodes
        nodes = []
        
        # Ego Node (Fixed position: Bottom Middle)
        ego_pos = (h - 1, w // 2)
        ego_depth = depth_norm[ego_pos[0], ego_pos[1]]
        # Use zeros for ego visual features or a generic placeholder
        nodes.append({
            "id": "ego",
            "type": "ego",
            "pos": [ego_pos[1] / w, ego_pos[0] / h],
            "depth": float(ego_depth),
            "features": np.zeros(384).tolist(), # Ego visual features placeholder
            "class": -1
        })
        
        object_visual_features = []
        
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                m_np = mask.cpu().numpy() # [mask_h, mask_w] - usually downsampled by YOLO
                m_resized = cv2.resize(m_np, (w, h))
                m_binary = (m_resized > 0.5).astype(np.uint8)
                
                if np.sum(m_binary) < 100: continue # Skip tiny objects
                
                # Centroid
                M = cv2.moments(m_binary)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Average Depth
                avg_depth = np.mean(depth_norm[m_binary == 1])
                
                # DINOv2 Token Pooling
                # Map mask to grid
                m_token_grid = cv2.resize(m_binary, (PATCH_GRID_SIZE, PATCH_GRID_SIZE), interpolation=cv2.INTER_AREA)
                m_token_binary = m_token_grid > 0.1 # Threshold for patch inclusion
                
                selected_tokens = patch_tokens[m_token_binary]
                if selected_tokens.shape[0] > 0:
                    node_visual_features = torch.mean(selected_tokens, dim=0).cpu().numpy()
                else:
                    # Fallback to the token at the centroid
                    tx = int((cx / w) * PATCH_GRID_SIZE)
                    ty = int((cy / h) * PATCH_GRID_SIZE)
                    tx = min(PATCH_GRID_SIZE - 1, max(0, tx))
                    ty = min(PATCH_GRID_SIZE - 1, max(0, ty))
                    node_visual_features = patch_tokens[ty, tx].cpu().numpy()
                
                cls_id = int(result.boxes[i].cls[0])
                label = result.names[cls_id]
                
                nodes.append({
                    "id": f"obj_{i}",
                    "type": "object",
                    "label": label,
                    "class": cls_id,
                    "pos": [cx / w, cy / h],
                    "depth": float(avg_depth),
                    "features": node_visual_features.tolist(),
                    "area": float(np.sum(m_binary) / (w * h))
                })
                object_visual_features.append(node_visual_features)

        # 6. Build Edges (Relative to Ego)
        edges = []
        ego_node = nodes[0]
        for node in nodes[1:]:
            # Relative spatial calcs
            dx = node["pos"][0] - ego_node["pos"][0]
            dy = node["pos"][1] - ego_node["pos"][1]
            dist = np.sqrt(dx**2 + dy**2)
            d_depth = node["depth"] - ego_node["depth"]
            bearing = np.arctan2(dx, -dy) # Relative to "up" which is forward
            
            edges.append({
                "source": node["id"],
                "target": "ego",
                "dist": float(dist),
                "d_depth": float(d_depth),
                "bearing": float(bearing)
            })

        # 7. Generate Pooled Embedding (Pipeline Compatibility)
        if object_visual_features:
            mean_obj_features = np.mean(object_visual_features, axis=0)
            # Add some spatial summary
            # [Mean_Visual (384) | Num_Objects (1) | Avg_Dist (1) | Avg_Depth (1)]
            num_objs = len(object_visual_features)
            avg_obj_dist = np.mean([e["dist"] for e in edges])
            avg_obj_depth = np.mean([n["depth"] for n in nodes[1:]])
            
            spatial_summary = np.array([num_objs / 20.0, avg_obj_dist, avg_obj_depth], dtype=np.float32)
            final_embedding = np.concatenate([mean_obj_features, spatial_summary])
        else:
            final_embedding = np.zeros(387, dtype=np.float32)
            
        # 8. Save Debug Info
        if "debug_output_path" in kwargs:
            # Save Graph JSON
            graph_data = {"nodes": nodes, "edges": edges}
            import json
            json_path = kwargs["debug_output_path"].replace(".jpg", "_graph.json")
            with open(json_path, "w") as f:
                json.dump(graph_data, f, indent=2)
                
            # Visualization
            vis_img = img_cv2.copy()
            
            # Draw Masks (Unique Colors)
            if result.masks is not None:
                mask_overlay = np.zeros_like(vis_img)
                for j, mask in enumerate(result.masks.data):
                    m_np = mask.cpu().numpy()
                    m_resized = cv2.resize(m_np, (w, h))
                    m_binary = (m_resized > 0.5).astype(np.uint8)
                    
                    # Generate a unique color for each object using a hash or predefined colormap
                    # We'll use a simple deterministic color generation based on index
                    color = [(j * 40) % 255, (j * 80) % 255, (j * 120) % 255]
                    mask_overlay[m_binary == 1] = color
                
                # Blend overlay with original image
                cv2.addWeighted(mask_overlay, 0.4, vis_img, 0.8, 0, vis_img)
            
            # Draw objects and Ego connection
            ego_px = (int(ego_node["pos"][0] * w), int(ego_node["pos"][1] * h))
            
            # Draw Edges first (so they are below nodes)
            # Find min/max distance for normalization
            dists = [e["dist"] for e in edges]
            max_dist = max(dists) if dists else 1.0
            min_dist = min(dists) if dists else 0.0
            
            for i, node in enumerate(nodes[1:]):
                obj_px = (int(node["pos"][0] * w), int(node["pos"][1] * h))
                edge = edges[i]
                
                # Thickness: Inversely proportional to distance (closer = thicker)
                # Normalize dist to [0, 1]
                norm_dist = (edge["dist"] - min_dist) / (max_dist - min_dist + 1e-8)
                thickness = int(max(1, (1.0 - norm_dist) * 8))
                
                # Color: Gradient based on depth (Blue = Close, Red = Far)
                # Depth Anything V2 normalized depth is 0 (close) to 1 (far)
                # BGR: (B, G, R)
                # Blue (255, 0, 0) -> Red (0, 0, 255)
                d = node["depth"]
                edge_color = (int((1.0 - d) * 255), int((1.0 - d) * 128), int(d * 255))
                
                # Line to Ego
                cv2.line(vis_img, obj_px, ego_px, edge_color, thickness)

            cv2.circle(vis_img, ego_px, 12, (0, 0, 255), -1) # Bold Red Ego
            cv2.circle(vis_img, ego_px, 14, (255, 255, 255), 2) # White Ring for Ego
            
            for i, node in enumerate(nodes[1:]):
                obj_px = (int(node["pos"][0] * w), int(node["pos"][1] * h))
                # Identify object index from id
                try:
                    j = int(node["id"].split("_")[1])
                    color = [(j * 40) % 255, (j * 80) % 255, (j * 120) % 255]
                except:
                    color = (0, 255, 0)

                # Center Point
                cv2.circle(vis_img, obj_px, 6, color, -1)
                cv2.circle(vis_img, obj_px, 8, (255, 255, 255), 1) # Small white ring
                
                # Line to Ego
                cv2.line(vis_img, obj_px, ego_px, (255, 255, 255), 1)
                
                # Label with background for readability
                label_str = f"{node['label']} (D:{node['depth']:.2f})"
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(vis_img, (obj_px[0]+10, obj_px[1]-th-2), (obj_px[0]+10+tw, obj_px[1]+2), (0,0,0), -1)
                cv2.putText(vis_img, label_str, (obj_px[0]+10, obj_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            cv2.imwrite(kwargs["debug_output_path"], vis_img)

        # Normalize and return
        final_embedding = final_embedding / (np.linalg.norm(final_embedding) + 1e-8)
        return final_embedding