import os
import torch
from typing import List, Tuple

class ModelLoader:
    """
    Loads models from a directory and extracts metadata from filenames.
    
    Attributes:
        model_dir (str): Directory containing the model files.
        model_paths (List[str]): List of paths to model files.
    """
    def __init__(self, model_dir: str):
        """
        Initializes the ModelLoader.
        
        Args:
            model_dir (str): Directory where models are stored.
        """
        self.model_dir = model_dir
        self.model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pth")]

    def get_model(self, path: str, device: torch.device) -> torch.nn.Module:
        """
        Loads a model from a given path and sets it to evaluation mode.
        
        Args:
            path (str): Path to the model file.
            device (torch.device): Device to load the model onto.
        
        Returns:
            torch.nn.Module: Loaded model.
        """
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        return model.to(device)
    
    def parse_model_name(self, model_name: str) -> Tuple[int, str, str]:
        """
        Parses the model filename to extract pruning ratio, architecture, and pruning distribution.
        
        Args:
            model_name (str): Filename of the model.
        
        Returns:
            Tuple[int, str, str]: Pruning ratio, architecture name, and pruning distribution.
        """
        parts = os.path.basename(model_name).split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected model filename format: {model_name}")
        arch = parts[0]
        if "UNPRUNED" in parts:
            return 0, arch, "UNPRUNED"
        pruning_distribution = next((p for p in parts if "PD" in p), "N/A")
        gpr = next((int(p.split("-")[1]) for p in parts if "GPR" in p), 0)
        return gpr, arch, pruning_distribution
