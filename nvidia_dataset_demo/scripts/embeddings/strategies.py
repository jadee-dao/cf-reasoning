import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import numpy as np
import cv2
from .base import EmbeddingStrategy
from .prompts import SCENE_DESCRIPTION_PROMPT, HAZARD_IDENTIFICATION_PROMPT

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


class FastVLMDescriptionStrategy(EmbeddingStrategy):
    """
    Prompts FastVLM to describe the scene in a structured format, 
    then embeds the generated text using SBERT.
    """
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, prompt_text=None):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.sbert = None
        self.model_loaded = False
        self.device = get_device()
        self.device = get_device()
        self.prompt_text = prompt_text or SCENE_DESCRIPTION_PROMPT

    def get_config_name(self):
        return "fastvlm_1.5b_description"

    def load_model(self):
        if not self.model_loaded:
            # 1. Load SBERT
            print(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
            
            # 2. Load FastVLM
            model_id = "apple/FastVLM-1.5B"
            print(f"Loading FastVLM model: {model_id}...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
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

    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        self.load_model()
        
        # 1. Load Image (Handle Video)
        if image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
             # Extract first frame
             cap = cv2.VideoCapture(image_path)
             ret, frame = cap.read()
             cap.release()
             if ret:
                 pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 raise ValueError(f"Could not read first frame from video: {image_path}")
        else:
             pil_image = Image.open(image_path).convert("RGB")
        
        # 2. Construct Prompt (Standard HF Usage)
        messages = [
            {"role": "user", "content": "<image>\n" + self.prompt_text}
        ]
        
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # 3. Tokenize and Splice Image Token
        pre, post = rendered.split("<image>", 1)
        
        # Tokenize text around the image token
        pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Splice in the IMAGE token id 
        img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        
        # 4. Preprocess Image
        # Use the logic proven to work with this model (generating 'pixel_values' specifically)
        # Note: We rely on the model's image processor to handle resizing.
        # HF 'images' arg in generate usually expects pixel_values if processed, or raw images if processor is handled internally
        # But FastVLM code snippet uses 'images=px' where px is pixel_values.
        px = self.model.get_vision_tower().image_processor(images=pil_image, return_tensors="pt")["pixel_values"]
        px = px.to(self.model.device, dtype=self.model.dtype)
        
        # 5. Generate
        with torch.no_grad():
            out = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=200,
            )
        
        generated_ids = out[0][input_ids.shape[1]:]
        description = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"\n[FastVLM Description]:\n{description}\n")

        # 6. Save Debug Info
        if "debug_output_path" in kwargs:
            txt_path = kwargs["debug_output_path"].replace(".jpg", ".txt")
            with open(txt_path, "w") as f:
                f.write(description)
                
            # Copy image for reference
            pil_image.save(kwargs["debug_output_path"])

        # 7. Embed with SBERT
        return self.sbert.encode(description)

class FastVLMHazardStrategy(FastVLMDescriptionStrategy):
    def __init__(self):
        super().__init__(prompt_text=HAZARD_IDENTIFICATION_PROMPT)

    def get_config_name(self):
        return "fastvlm_1.5b_hazard"

# --- OpenRouter Strategies ---

import requests
import base64
import os

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
            
            # Save the stitched grid!
            cv2.imwrite(kwargs["debug_output_path"], grid_img)

        # 7. Embed
        return self.sbert.encode(description)