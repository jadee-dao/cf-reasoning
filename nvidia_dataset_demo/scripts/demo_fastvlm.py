import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_model():
    print("Loading model...")
    model_id = "apple/FastVLM-0.5B" 
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Try to load processor, if fails we might need to use model's internal processing
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except:
            processor = None
            print("AutoProcessor not found, will check model config for image processing.")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    print("Model loaded successfully.")
    return model, tokenizer, processor

def get_first_frame(video_path):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    else:
        print("Failed to read video frame.")
        return None

def process_image(model, tokenizer, image):
    # Depending on how FastVLM is implemented, we might need specific preprocessing.
    # Looking at standard LLaVA/FastVLM usage: usually pixel_values are needed.
    
    # Check if model has a vision tower with image processor
    vision_tower = model.model.vision_tower
    
    # Access the image processor from the vision tower if available, standard in LLaVA
    if hasattr(vision_tower, 'image_processor'):
        image_processor = vision_tower.image_processor
    # Fallback or manual access if structure differs
    elif hasattr(model.config, 'vision_config'):
        # Create a processor based on config if needed, but for now lets assume standard usage
        pass

    # For FastVLM specifically, it uses specific preprocessing usually.
    # Let's try to infer from the model's forward signature or 'prepare_inputs_labels_for_multimodal'
    
    # We will try to invoke the vision tower directly to get features for "Importance"
    # The vision tower input usually expects [Batch, 3, H, W]
    
    # Standard transforms for CLIP/SigLIP/FastViT: Resize -> Tensor -> Normalize
    # We'll use the model's internal image processor if we can find it.
    
    # Hack: let's inspect the vision_tower class to see if it has a preprocess
    
    # If we can't find a processor easily, we might fail. 
    # But usually 'transformers' AutoProcessor handles it if it exists. 
    # If processor is None, we need to handle it.
    
    return image # Placeholder if we do manual processing inside extraction

def extract_importance_and_embedding(model, tokenizer, pil_image):
    # 1. Preprocess Image
    # We need to turn PIL image into tensor matching model expectation
    # FastVLM uses FastViT. It likely expects 256x256 or 384x384.
    
    # Let's look for image_processor in the loaded components
    image_processor = None
    if hasattr(model.model.vision_tower, 'image_processor'):
        image_processor = model.model.vision_tower.image_processor
    
    if image_processor:
        inputs = image_processor(pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(model.device)
        print(f"Image processed to shape: {pixel_values.shape}")
    else:
        print("Could not find image processor. Attempting default CLIP-style transform.")
        # Fallback transform (Resize to 384? check config)
        # FastVLM often uses 384x384
        try:
            from transformers import CLIPImageProcessor
            proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336") # Random guess for similar size
            inputs = proc(pil_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(model.device)
        except:
             print("Fallback failed.")
             return

    # 2. Forward pass through Vision Tower
    with torch.no_grad():
        # vision_tower(pixel_values) usually returns features or (features, hidden_states)
        # We need to inspect what it returns.
        vision_outputs = model.model.vision_tower(pixel_values)
        
        # If it returns a tuple, usually 0 is the last hidden state
        if isinstance(vision_outputs, tuple):
            image_features = vision_outputs[0]
        else:
            image_features = vision_outputs
            
        print(f"Vision Tower Output Shape: {image_features.shape}")
        
    # image_features shape: [Batch, SeqLen, Dim] or [Batch, Dim, H, W] depending on model
    # FastViT likely returns [Batch, Dim, H, W] from the backbone, but the wrapper might flatten it.
    
    # If shape is [B, L, D] (Sequence), we need to know H, W to reshape for heatmap
    # If shape is [B, D, H, W], we can average D directly.
    
    importance_map = None
    embedding = None
    
    if len(image_features.shape) == 4: # [B, C, H, W]
        # Average across channels (C)
        activation_map = torch.mean(image_features, dim=1) # [B, H, W]
        importance_map = activation_map[0].cpu().numpy()
        
        # Embedding: Global Average Pooling
        embedding = torch.mean(image_features, dim=(2, 3)) # [B, C]
        
    elif len(image_features.shape) == 3: # [B, L, D]
        # It's a sequence. L = H*W (usually)
        # We need to infer H and W. Sqrt(L). 
        # FastVLM 0.5B: L=256? 
        B, L, D = image_features.shape
        H = W = int(L**0.5)
        if H*W != L:
            print(f"Warning: Sequence length {L} is not a perfect square. Cannot easily reshape to image.")
            # Might be [CLS] token + Patches
            # Try L-1
            if int((L-1)**0.5)**2 == L-1:
                 H = W = int((L-1)**0.5)
                 # Ignore CLS
                 features_spatial = image_features[:, 1:, :] 
                 activation_map = torch.mean(features_spatial, dim=2) # [B, L-1]
                 activation_map = activation_map.reshape(B, H, W)
                 importance_map = activation_map[0].cpu().numpy()
            else:
                 # Just average magnitude for embedding
                 pass
        else:
            activation_map = torch.mean(image_features, dim=2) # [B, L]
            activation_map = activation_map.reshape(B, H, W)
            importance_map = activation_map[0].cpu().numpy()
            
        # Embedding: Mean of all tokens
        embedding = torch.mean(image_features, dim=1) # [B, D]
        
    # Save results
    if importance_map is not None:
        # Resize to original image size for overlay
        # Normalize to 0-1
        importance_map_norm = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())
        
        # Resize importance map to image size using cv2
        img_np = np.array(pil_image)
        heatmap_resized = cv2.resize(importance_map_norm, (img_np.shape[1], img_np.shape[0]))
        
        # Overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(pil_image)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(overlay)
        plt.title("FastVLM 'Importance'")
        plt.axis('off')
        
        # --- Importance Filtering ---
        # Threshold: Keep top 30% most important pixels? Or > Mean?
        threshold = np.mean(heatmap_resized) + 0.5 * np.std(heatmap_resized)
        mask = heatmap_resized > threshold
        
        # Create masked image (black out unimportant)
        masked_img_np = img_np.copy()
        # Option A: Black out
        masked_img_np[~mask] = 0 
        # Option B: Blur (so context is vaguely there)? Let's stick to black for strict attention.
        
        masked_pil = Image.fromarray(masked_img_np)
        
        plt.subplot(1, 4, 3)
        plt.imshow(masked_pil)
        plt.title(f"Masked (>{threshold:.2f})")
        plt.axis('off')
        
        output_file = "nvidia_dataset_demo/scripts/analysis_results/fastvlm_importance_process.png"
        plt.savefig(output_file)
        print(f"Saved process visualization to {output_file}")
        
        # --- Re-Embedding ---
        print("Extracting focused embedding from masked image...")
        focused_embedding = extract_embedding_only(model, tokenizer, masked_pil)
        
        return embedding, focused_embedding

    return embedding, None

def extract_embedding_only(model, tokenizer, pil_image):
    # Reuse extraction logic (simplified)
    image_processor = None
    if hasattr(model.model.vision_tower, 'image_processor'):
        image_processor = model.model.vision_tower.image_processor
    
    encoded_image = None
    if image_processor:
        inputs = image_processor(pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(model.device)
    else:
         return None

    with torch.no_grad():
        vision_outputs = model.model.vision_tower(pixel_values)
        if isinstance(vision_outputs, tuple):
            image_features = vision_outputs[0]
        else:
            image_features = vision_outputs
            
        if len(image_features.shape) == 4:
            embedding = torch.mean(image_features, dim=(2, 3)) 
        elif len(image_features.shape) == 3:
            embedding = torch.mean(image_features, dim=1)
            
    return embedding

def main():
    model, tokenizer, processor = load_model()
    if not model:
        return

    # Find a video
    video_dir = "nvidia_dataset_demo/extracted_data"
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not video_files:
        print("No videos found.")
        return
        
    video_path = os.path.join(video_dir, video_files[0])
    print(f"Processing {video_path}...")
    
    image = get_first_frame(video_path)
    if image:
        embedding, focused_embedding = extract_importance_and_embedding(model, tokenizer, image)
        
        if embedding is not None and focused_embedding is not None:
             print(f"Original Embedding Shape: {embedding.shape}")
             print(f"Focused Embedding Shape: {focused_embedding.shape}")

if __name__ == "__main__":
    main()
