import pynvml

class EnergyMonitor:
    """
    Monitors GPU energy consumption using NVIDIA's NVML library.
    
    Attributes:
        gpu_index (int): Index of the GPU being monitored.
        device_handle (pynvml.nvmlDevice_t): Handle to the GPU device for querying energy consumption.
    """

    def __init__(self, gpu_index: int = 0):
        """
        Initializes the EnergyMonitor with a specific GPU index.
        
        Args:
            gpu_index (int): Index of the GPU to monitor (default: 0).
        
        Raises:
            RuntimeError: If the GPU index is invalid or NVML initialization fails.
        """
        try:
            pynvml.nvmlInit()
            self.gpu_index = gpu_index
            self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML: {str(e)}")

    def get_energy(self) -> float:
        """
        Returns the total energy consumption of the GPU in Joules.
        
        Returns:
            float: Energy consumption in Joules.
        
        Raises:
            RuntimeError: If energy consumption retrieval fails.
        """
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.device_handle) / 1000  # Convert mJ to J
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to get energy consumption: {str(e)}")

    def close(self):
        """
        Shuts down NVML to free resources.
        """
        pynvml.nvmlShutdown()
