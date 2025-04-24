import torch
import statistics
from typing import Tuple
from tqdm import tqdm
from .energy_monitor import EnergyMonitor

class InferenceRunner:
    """
    Runs inference on a model and measures execution time and energy consumption.

    Attributes:
        model (torch.nn.Module): Model to be evaluated.
        device (torch.device): Device for computation.
        num_iters (int): Number of inferences to perform per trial.
        num_trials (int): Number of repetitions of the experiment.
        energy_monitor (EnergyMonitor): Utility to measure GPU energy.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, num_iters: int, num_trials: int):
        self.model = model
        self.device = device
        self.num_iters = num_iters
        self.num_trials = num_trials
        self.energy_monitor = EnergyMonitor()

    def run(self, input_tensor: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Executes inference multiple times and returns timing and energy statistics.

        Args:
            input_tensor (torch.Tensor): The input data for the model.

        Returns:
            Tuple[float, float, float, float]: Mean and std of inference time and energy per sample.
        """
        times = []
        energies = []

        for trial in range(self.num_trials):
            # Warm-up
            with torch.no_grad():
                for _ in range(100):
                    self.model(input_tensor)
            torch.cuda.synchronize()
            batch_size = input_tensor.shape[0]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_energy = self.energy_monitor.get_energy()
            

            start_event.record()
            with torch.no_grad():
                for _ in tqdm(range(self.num_iters), desc=f"Trial {trial + 1}/{self.num_trials}: Running {self.num_iters} iterations", leave=True):
                    self.model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()

            end_energy = self.energy_monitor.get_energy()

            elapsed_time = start_event.elapsed_time(end_event) / 1000  # ms to s
            energy_used = end_energy - start_energy

            time_per_sample = elapsed_time / self.num_iters
            energy_per_sample = energy_used / self.num_iters

            times.append(time_per_sample / batch_size)
            energies.append(energy_per_sample / batch_size)

            # Clear cache at the end of each trial
            torch.cuda.empty_cache()

        mean_time = sum(times) / len(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        mean_energy = sum(energies) / len(energies)
        std_energy = statistics.stdev(energies) if len(energies) > 1 else 0

        return mean_time, std_time, mean_energy, std_energy