from abc import ABC, abstractmethod
import numpy as np

class EmbeddingStrategy(ABC):
    """
    Abstract Base Class for embedding generation strategies.
    
    Subclasses must implement:
        - load_model(): Load the specific model(s) needed.
        - generate_embedding(data_dict): Produce a 1D numpy array embedding.
    """
    
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self):
        """Loads the model and any necessary processors."""
        pass
        
    @abstractmethod
    def generate_embedding(self, image_path: str, **kwargs) -> np.array:
        """
        Generates an embedding for the given data.
        
        Args:
            image_path (str): Path to the image frame (e.g., middle frame).
            **kwargs: Additional data like 'yolo_results', 'frame_index', etc.
            
        Returns:
            np.array: A 1D numpy array representing the embedding.
        """
        pass
