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
        energy_monitor (EnergyMonitor): Monitors GPU energy consumption.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, num_iters: int):
        """
        Initializes the InferenceRunner.
        
        Args:
            model (torch.nn.Module): Model to run inference on.
            device (torch.device): Device to use for inference.
            num_iters (int): Number of iterations to measure.
        """
        self.model = model
        self.device = device
        self.num_iters = num_iters
        self.energy_monitor = EnergyMonitor()
    
    def run(self, input_tensor: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Executes multiple inference iterations and returns timing and energy statistics.
        
        Args:
            input_tensor (torch.Tensor): Input tensor for the model.
        
        Returns:
            Tuple[float, float, float, float]: Mean time per sample, std time per sample, mean energy, std energy.
        """
        times, energies = [], []
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        batch_size = input_tensor.shape[0]
        
        for _ in tqdm(range(self.num_iters), desc=f"Running {self.num_iters} iterations"):
            start_energy = self.energy_monitor.get_energy()
            start_time.record()
            with torch.no_grad():
                self.model(input_tensor)
            end_time.record()
            torch.cuda.synchronize()
            end_energy = self.energy_monitor.get_energy()

            times.append((start_time.elapsed_time(end_time) / 1000) / batch_size)  # Convert to seconds and normalize
            energies.append(end_energy - start_energy)

        mean_time = sum(times) / len(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        mean_energy = sum(energies) / len(energies)
        std_energy = statistics.stdev(energies) if len(energies) > 1 else 0
        return mean_time, std_time, mean_energy, std_energy
