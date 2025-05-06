import torch
import pandas as pd
from tqdm import tqdm
from typing import List

from .model_loader import ModelLoader
from .model_analyzer import ModelAnalyzer
from .inference_runner import InferenceRunner
from .result_saver import ResultSaver
import time

class ExperimentRunner:
    """
    Manages the execution of inference experiments across multiple models and batch sizes.

    Attributes:
        model_loader (ModelLoader): Loads models from a directory.
        batch_sizes (List[int]): List of batch sizes for experimentation.
        num_trials (int): Number of trials per experiment.
        num_iters (int): Number of inference iterations per trial.
        device (torch.device): Device to run inference on.
        result_saver (ResultSaver): Saves experiment results.
        input_channels (int): Number of input channels for the tensor.
        input_height (int): Height of the input tensor.
        input_width (int): Width of the input tensor.
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
        Initializes the ExperimentRunner.

        Args:
            model_dir (str): Directory containing models for analysis.
            batch_sizes (List[int]): Batch sizes to test.
            num_trials (int): Number of trials for each configuration.
            num_iters (int): Number of iterations per inference run.
            input_channels (int): Number of input channels.
            input_height (int): Height of the input tensor.
            input_width (int): Width of the input tensor.
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
        Runs the experiment for all models and batch sizes.

        Returns:
            pd.DataFrame: Dataframe containing the results of the experiments.
        """
        results = []

        for model_path in tqdm(self.model_loader.model_paths, desc="Processing Models"):
            print(f"Processing model: {model_path}")
            model = self.model_loader.get_model(model_path, self.device)
            gpr, arch, pruning_distribution = self.model_loader.parse_model_name(model_path)

            for batch_size in self.batch_sizes:
               
                print(f"Processing batch size: {batch_size}")
                if self.model_loader.skip_completed and self.model_loader.is_experiment_completed(gpr, arch, pruning_distribution, batch_size):
                    print(f"Skipping completed experiment for: {gpr}, {arch}, {pruning_distribution}, batch size {batch_size}")
                    continue
                input_tensor = torch.randn(batch_size, self.input_channels, self.input_height, self.input_width).to(self.device)
                flops, params = ModelAnalyzer.analyze(model, input_tensor)
                inference_runner = InferenceRunner(model, self.device, self.num_iters, self.num_trials)

                mean_time, std_time, mean_energy, std_energy = inference_runner.run(input_tensor)
                fps = 1.0 / mean_time if mean_time > 0 else float('inf')

                result = {
                    "GPR": gpr,
                    "Architecture": arch,
                    "Pruning Distribution": pruning_distribution,
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
                #time.sleep(60)

        return pd.DataFrame(results)

    def __del__(self):
        """Ensures NVML is properly shutdown when the experiment is finished."""
        import pynvml
        pynvml.nvmlShutdown()



