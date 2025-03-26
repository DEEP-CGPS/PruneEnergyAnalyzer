import os
import csv
from typing import Dict

class ResultSaver:
    """
    Saves experiment results to a CSV file.
    
    Attributes:
        filename (str): Path to the CSV file where results will be saved.
        file_exists (bool): Indicates whether the file already exists to avoid rewriting headers.
    """
    
    def __init__(self, filename: str = "experiment_results.csv"):
        """
        Initializes the ResultSaver.
        
        Args:
            filename (str): Name of the CSV file to save results.
        """
        self.filename = filename
        self.file_exists = os.path.exists(self.filename)
    
    def save(self, result: Dict[str, float]) -> None:
        """
        Appends a result entry to the CSV file.
        
        Args:
            result (Dict[str, float]): A dictionary containing experimental results.
        """
        write_header = not self.file_exists
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        self.file_exists = True
