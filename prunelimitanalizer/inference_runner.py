import torch
import statistics
from tqdm import tqdm
from .energy_monitor import EnergyMonitor
from typing import Tuple

class InferenceRunner:
    """
    Runs inference on a model and measures execution time and energy consumption.
    
    Attributes:
        model (torch.nn.Module): Model to run inference on.
        device (torch.device): Device to run inference on.
        num_iters (int): Number of inference iterations.
        num_trials (int): Number of trials for statistical analysis.
        energy_monitor (EnergyMonitor): Monitors GPU energy consumption.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, num_iters: int, num_trials: int):
        """
        Initializes the InferenceRunner.
        
        Args:
            model (torch.nn.Module): Model to run inference on.
            device (torch.device): Device to use for inference.
            num_iters (int): Number of iterations to measure.
            num_trials (int): Number of trials for statistical analysis.
        """
        self.model = model
        self.device = device
        self.num_iters = num_iters
        self.num_trials = num_trials
        self.energy_monitor = EnergyMonitor()
    
    def run(self, input_tensor: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Executes multiple inference trials and returns timing and energy statistics.
        
        Args:
            input_tensor (torch.Tensor): Input tensor for the model.
        
        Returns:
            Tuple[float, float, float, float]: Mean time per sample, std time per sample, mean energy per sample, std energy per sample.
        """
        times, energies = [], []
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        batch_size = input_tensor.shape[0]
        
        for trial in range(self.num_trials):
            trial_times, trial_energies = [], []
            for _ in tqdm(range(self.num_iters), desc=f"Trial {trial + 1}/{self.num_trials}: Running {self.num_iters} iterations"):
                start_energy = self.energy_monitor.get_energy()
                start_time.record()
                with torch.no_grad():
                    self.model(input_tensor)
                end_time.record()
                torch.cuda.synchronize()
                end_energy = self.energy_monitor.get_energy()

                trial_times.append((start_time.elapsed_time(end_time) / 1000))  # Convert to seconds and normalize
                trial_energies.append((end_energy - start_energy))  # Normalize energy per sample
            
            times.append((sum(trial_times) / len(trial_times)) / batch_size)
            energies.append((sum(trial_energies) / len(trial_energies)) / batch_size)

        mean_time = sum(times) / len(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        mean_energy = sum(energies) / len(energies)
        std_energy = statistics.stdev(energies) if len(energies) > 1 else 0
        return mean_time, std_time, mean_energy, std_energy
