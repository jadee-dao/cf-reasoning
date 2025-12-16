import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import numpy as np
import cv2
from .base import EmbeddingStrategy

# --- Constants ---
SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
SBERT_MODEL_NAME = "all-mpnet-base-v2"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
YOLO_MODEL_PATH = "yolo11n.pt" # Or 'yolo11n-seg.pt' for segmentation

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ... (Naive, Foreground, TextDescription classes remain unchanged) ...

class TextDescriptionStrategy(EmbeddingStrategy):
    # ... (code for TextDescriptionStrategy) ...
    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        # ... (implementation as fixed previously) ...
        return self.model.encode(description)




class VideoXClipStrategy(EmbeddingStrategy):
    """
    Uses X-CLIP to embed the entire video clip by sampling 8 frames.
    Captures both visual semantics and temporal dynamics.
    """
    def load_model(self):
        if not self.model_loaded:
            model_name = "microsoft/xclip-base-patch32"
            print(f"Loading X-CLIP model: {model_name}...")
            self.device = get_device()
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        # We need the video path. 
        # If not provided in kwargs, try to infer it (hacky fallback), but run_embedding_test should provide it.
        video_path = kwargs.get("video_path")
        if not video_path:
            # Fallback: assuming image_path is like ".../temp_frame_ID.jpg" and video is ID.mp4? 
            # Ideally this shouldn't happen if we update the main script.
            raise ValueError("Video path not provided for Video Strategy")

        # 1. Sample 8 Frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # uniform sampling
        indices = np.linspace(0, total_frames-1, 8).astype(int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # Pad with last frame or black if fail
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        
        # 2. Prepare Debug Filmstrip
        if "debug_output_path" in kwargs:
            # Create a 2x4 grid or 1x8 strip
            # Let's do 1x8 strip
            # frames[0] might be big. Resize to height 224 for strip?
            target_h = 224
            scale = target_h / frames[0].shape[0]
            target_w = int(frames[0].shape[1] * scale)
            
            resized_frames = [cv2.resize(f, (target_w, target_h)) for f in frames]
            filmstrip = np.concatenate(resized_frames, axis=1)
            
            # Save
            # Convert to BGR for cv2 save or use PIL
            Image.fromarray(filmstrip).save(kwargs["debug_output_path"])

        # 3. Generate Embedding
        # X-CLIP expects video pixel values. Processor handles it.
        inputs = self.processor(videos=list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            video_features = self.model.get_video_features(**inputs)
            
        # Normalize
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
        return video_features.cpu().numpy().flatten()

class VLMCaptionStrategy(EmbeddingStrategy):
    """
    Uses a VLM (BLIP) to generate a natural language caption for the image,
    then embeds that caption using SBERT.
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
            
            print(f"Loading BLIP model: {BLIP_MODEL_NAME}...")
            self.device = get_device()
            self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(self.device)
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        image = Image.open(image_path).convert("RGB")
        
        # 1. Generate Caption
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate with slightly tweaked parameters for better description
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # 2. Save Debug Info
        if "debug_output_path" in kwargs:
            # For VLM, the "debug image" is just the original image, 
            # but we definitely want to save the text.
            image.save(kwargs["debug_output_path"])
            
            txt_path = kwargs["debug_output_path"].replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.write(caption)

        # 3. Embed Caption
        return self.sbert.encode(caption)

class NaiveStrategy(EmbeddingStrategy):
    """
    Computes embedding of the full image using SigLIP.
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SigLIP model: {SIGLIP_MODEL_NAME}...")
            self.device = get_device()
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            self.model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device)
            self.model_loaded = True
            
    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        image = Image.open(image_path).convert("RGB")
        
        if "debug_output_path" in kwargs:
            image.save(kwargs["debug_output_path"])
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
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


class TextDescriptionStrategy(EmbeddingStrategy):
    """
    Uses YOLO to detect objects and their positions, templates a descriptive sentence,
    and computes the text embedding using SBERT.
    """
    def load_model(self):
        if not self.model_loaded:
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            self.model = SentenceTransformer(SBERT_MODEL_NAME)
            
            print(f"Loading YOLO Detection model: {YOLO_MODEL_PATH}...")
            self.det_model = YOLO("yolo11n.pt")
            self.model_loaded = True

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        # 1. Run Object Detection
        results = self.det_model(image_path, verbose=False)
        result = results[0]
        
        width = result.orig_shape[1]
        
        # 2. Bin objects spatially
        left_objs = []
        center_objs = []
        right_objs = []
        
        # Define thresholds for Left/Center/Right (33% split)
        thr1 = width / 3
        thr2 = 2 * width / 3
        
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                
                if cx < thr1:
                    left_objs.append(label)
                elif cx < thr2:
                    center_objs.append(label)
                else:
                    right_objs.append(label)
        
        # 3. Helper to format list: "2 cars, 1 person"
        def format_counts(obj_list):
            if not obj_list:
                return "empty"
            counts = {}
            for obj in obj_list:
                counts[obj] = counts.get(obj, 0) + 1
            return ", ".join([f"{c} {o}{'s' if c>1 else ''}" for o, c in counts.items()])
            
        left_str = format_counts(left_objs)
        center_str = format_counts(center_objs)
        right_str = format_counts(right_objs)
        
        # 4. Template String
        description = f"A driving scene. Left: {left_str}. Center: {center_str}. Right: {right_str}."
        
        # 5. Save Debug Info
        if "debug_output_path" in kwargs:
            # Save annotated image for text strategy
            res_plotted = result.plot()
            cv2.imwrite(kwargs["debug_output_path"], res_plotted)
            
            # Save the exact text description used
            txt_path = kwargs["debug_output_path"].replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.write(description)
        
        # 6. Generate Embedding
        # SBERT encode expects a list or string
        embedding = self.model.encode(description)
        return embedding


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
