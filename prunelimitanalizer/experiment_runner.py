import torch
import pandas as pd
from tqdm import tqdm
from typing import List

from .model_loader import ModelLoader
from .model_analyzer import ModelAnalyzer
from .inference_runner import InferenceRunner
from .result_saver import ResultSaver


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
    """

    def __init__(self, model_dir: str, batch_sizes: List[int], num_trials: int = 10, num_iters: int = 50):
        """
        Initializes the ExperimentRunner.

        Args:
            model_dir (str): Directory containing models for analysis.
            batch_sizes (List[int]): Batch sizes to test.
            num_trials (int): Number of trials for each configuration.
            num_iters (int): Number of iterations per inference run.
        """
        self.model_loader = ModelLoader(model_dir)
        self.batch_sizes = batch_sizes
        self.num_trials = num_trials
        self.num_iters = num_iters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_saver = ResultSaver()

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
                input_tensor = torch.randn(batch_size, 3, 224, 224).to(self.device)
                flops, params = ModelAnalyzer.analyze(model, input_tensor)
                inference_runner = InferenceRunner(model, self.device, self.num_iters, self.num_trials)

                mean_time, std_time, mean_energy, std_energy = inference_runner.run(input_tensor)

                result = {
                    "GPR": gpr,
                    "Architecture": arch,
                    "Pruning Distribution": pruning_distribution,
                    "BATCH_SIZE": batch_size,
                    "Mean Time per Sample (s)": mean_time,
                    "STD Time per Sample (s)": std_time,
                    "Mean Energy per Sample (J)": mean_energy,
                    "STD Energy per Sample (J)": std_energy,
                    "Parameters": params,
                    "FLOPs": flops,
                }

                results.append(result)
                self.result_saver.save(result)

        return pd.DataFrame(results)

    def __del__(self):
        """Ensures NVML is properly shutdown when the experiment is finished."""
        import pynvml
        pynvml.nvmlShutdown()
