import os
import torch
import random
import pandas as pd
from typing import List, Tuple, Optional

class ModelLoader:
    """
    Loads models from a directory and manages metadata and experiment completion status.

    Attributes:
        model_dir (str): Directory containing the model files.
        model_paths (List[str]): List of paths to model files.
        skip_completed (bool): Whether to skip already completed experiments.
        results_df (pd.DataFrame or None): Cached results to check if experiments are already completed.
    """

    def __init__(self, model_dir: str, result_file: Optional[str] = None, skip_completed: bool = True, shuffle=True):
        """
        Initializes the ModelLoader.

        Args:
            model_dir (str): Directory where models are stored.
            result_file (str, optional): Path to the CSV file with experiment results.
            skip_completed (bool): Whether to skip already completed experiments.
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

    def is_experiment_completed(self, gpr: int, arch: str, pruning_distribution: str, batch_size: int) -> bool:
        """
        Checks if an experiment with the given parameters has already been completed.

        Args:
            gpr (int): Global pruning ratio.
            arch (str): Model architecture.
            pruning_distribution (str): Pruning distribution.
            batch_size (int): Batch size.

        Returns:
            bool: True if the experiment has already been completed, False otherwise.
        """
        if self.results_df is None:
            return False

        match = (
            (self.results_df["GPR"] == gpr) &
            (self.results_df["Architecture"] == arch) &
            (self.results_df["Pruning Distribution"] == pruning_distribution) &
            (self.results_df["BATCH_SIZE"] == batch_size)
        )

        return match.any()

