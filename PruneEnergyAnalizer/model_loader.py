import os
import torch
import random
import pandas as pd
from typing import List, Tuple, Optional

class ModelLoader:
    """
    Loads models from a directory and manages metadata and experiment completion status.

    Attributes:
        model_dir (str): Directory containing model files.
        model_paths (List[str]): List of model file paths.
        skip_completed (bool): Flag to skip models already evaluated.
        result_file (Optional[str]): Path to the CSV file storing experiment results.
        results_df (Optional[pd.DataFrame]): DataFrame loaded from the result file.
    """

    def __init__(self, model_dir: str, result_file: Optional[str] = None, skip_completed: bool = True, shuffle=True):
        """
        Initializes the ModelLoader.

        Args:
            model_dir (str): Directory containing model files.
            result_file (Optional[str]): Path to the results CSV file.
            skip_completed (bool): If True, will skip models already evaluated.
            shuffle (bool): If True, will shuffle the model list.
        """
        self.model_dir = model_dir
        self.model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pth")]
        if shuffle:
            random.shuffle(self.model_paths)
        self.skip_completed = skip_completed
        self.result_file = result_file
        self.results_df = None

        if self.skip_completed and self.result_file and os.path.exists(self.result_file):
            try:
                self.results_df = pd.read_csv(self.result_file)
            except Exception as e:
                print(f"Warning: could not read results file: {e}")
                self.results_df = None

    def get_model(self, path: str, device: torch.device) -> torch.nn.Module:
        """
        Loads a model from a file path and moves it to the specified device.

        Args:
            path (str): Path to the model file.
            device (torch.device): Device to which the model will be moved.

        Returns:
            torch.nn.Module: Loaded model.
        """
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        return model.to(device)

    def is_experiment_completed(self, model_path: str, batch_size: int) -> bool:
        """
        Checks if an experiment with the given model path and batch size has already been completed.

        Args:
            model_path (str): Path to the model file.
            batch_size (int): Batch size used in the experiment.

        Returns:
            bool: True if the experiment has already been completed, False otherwise.
        """
        if self.results_df is None:
            return False

        model_name = os.path.basename(model_path)
        match = (
            (self.results_df["MODEL_NAME"] == model_name) &
            (self.results_df["BATCH_SIZE"] == batch_size)
        )
        return match.any()



