import torch
from fvcore.nn import FlopCountAnalysis
from typing import Tuple

class ModelAnalyzer:
    """
    Analyzes model complexity in terms of FLOPs and parameters.
    
    Methods:
        analyze(model, input_tensor): Computes FLOPs and the number of parameters.
    """
    
    @staticmethod
    def analyze(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[int, int]:
        """
        Computes the number of FLOPs and parameters for a given model.
        
        Args:
            model (torch.nn.Module): The model to analyze.
            input_tensor (torch.Tensor): Example input tensor to estimate FLOPs.
        
        Returns:
            Tuple[int, int]:
                - Total FLOPs (Floating Point Operations per Second) required for a forward pass.
                - Total number of trainable parameters in the model.
        """
        flops = FlopCountAnalysis(model, input_tensor).total()
        params = sum(p.numel() for p in model.parameters())
        return flops, params
