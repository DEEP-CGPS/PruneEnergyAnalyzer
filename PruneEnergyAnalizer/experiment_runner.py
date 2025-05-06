import torch
import pandas as pd
from tqdm import tqdm
from typing import List
import os

from .model_loader import ModelLoader
from .model_analyzer import ModelAnalyzer
from .inference_runner import InferenceRunner
from .result_saver import ResultSaver
import time

class ExperimentRunner:
    """
    Orchestrates the process of running inference experiments on a set of models.

    Attributes:
        model_loader (ModelLoader): Loads models and checks if experiments are completed.
        batch_sizes (List[int]): List of batch sizes to test.
        num_trials (int): Number of trials to run for each configuration.
        num_iters (int): Number of iterations per trial.
        device (torch.device): Device on which to run the experiments.
        result_saver (ResultSaver): Saves the results of experiments to CSV.
        input_channels (int): Number of input channels for the model input.
        input_height (int): Height of the model input.
        input_width (int): Width of the model input.
    """

    def __init__(
        self, 
        model_dir: str, 
        batch_sizes: List[int], 
        num_trials: int = 10, 
        num_iters: int = 50, 
        input_channels: int = 3, 
        input_height: int = 224, 
        input_width: int = 224, 
        filename: str = "experiment_results.csv",
        skip_completed: bool = True
    ):
        """
        Initializes the ExperimentRunner with the given parameters.

        Args:
            model_dir (str): Directory containing model files.
            batch_sizes (List[int]): List of batch sizes to use.
            num_trials (int): Number of trials per configuration.
            num_iters (int): Number of iterations per trial.
            input_channels (int): Channels in the input tensor.
            input_height (int): Height of the input tensor.
            input_width (int): Width of the input tensor.
            filename (str): CSV file to save results.
            skip_completed (bool): Whether to skip already completed experiments.
        """
        self.model_loader = ModelLoader(model_dir, result_file=filename, skip_completed=skip_completed)
        self.batch_sizes = batch_sizes
        self.num_trials = num_trials
        self.num_iters = num_iters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_saver = ResultSaver(filename)
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

    def run_experiment(self) -> pd.DataFrame:
        """
        Runs the inference experiment on all models and batch sizes.

        Returns:
            pd.DataFrame: DataFrame containing the results of all experiments.
        """
        results = []

        for model_path in tqdm(self.model_loader.model_paths, desc="Processing Models"):
            print(f"Processing model: {model_path}")
            model = self.model_loader.get_model(model_path, self.device)
            model_name = os.path.basename(model_path)

            for batch_size in self.batch_sizes:
                print(f"Processing batch size: {batch_size}")
                if self.model_loader.skip_completed and self.model_loader.is_experiment_completed(model_path, batch_size):
                    print(f"Skipping completed experiment for: {model_name}, batch size {batch_size}")
                    continue

                input_tensor = torch.randn(batch_size, self.input_channels, self.input_height, self.input_width).to(self.device)
                flops, params = ModelAnalyzer.analyze(model, input_tensor)
                inference_runner = InferenceRunner(model, self.device, self.num_iters, self.num_trials)

                mean_time, std_time, mean_energy, std_energy = inference_runner.run(input_tensor)
                fps = 1.0 / mean_time if mean_time > 0 else float('inf')

                result = {
                    "MODEL_NAME": model_name,
                    "BATCH_SIZE": batch_size,
                    "Mean Time per Sample (s)": mean_time,
                    "FPS": fps,
                    "STD Time per Sample (s)": std_time,
                    "Mean Energy per Sample (J)": mean_energy,
                    "STD Energy per Sample (J)": std_energy,
                    "Parameters": params,
                    "FLOPs": flops,
                }

                results.append(result)
                self.result_saver.save(result)
                # time.sleep(60)

        return pd.DataFrame(results)

    def __del__(self):
        """
        Ensures that NVML is properly shutdown after experiments are complete.
        """
        import pynvml
        pynvml.nvmlShutdown()



