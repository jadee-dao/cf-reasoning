from .classifier import BaselineVideoClassifier, BaselineImageClassifier

def create_model(modality: str, backbone: str, num_classes: int):
    """
    Factory function to create a model based on input modality.
    """
    print(f"Factory: Creating model for modality '{modality}' with backbone '{backbone}'...")
    
    if modality == 'video':
        # Video models (3D CNN)
        if backbone == 'default':
             backbone = 'r3d_18'
        return BaselineVideoClassifier(backbone_name=backbone, num_classes=num_classes)
        
    elif modality == 'image':
        # Image models (2D CNN)
        if backbone == 'default':
             backbone = 'resnet50' # Default image backbone
        return BaselineImageClassifier(backbone_name=backbone, num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown modality: {modality}. Supported: 'video', 'image'.")
