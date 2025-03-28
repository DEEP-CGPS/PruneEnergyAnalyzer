import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class AnalysisPlotter:
    """
    Class for analyzing and visualizing the impact of pruning on energy consumption.
    """
    def __init__(self, dataframe: pd.DataFrame, x_column: str, y_column: str, title: str):
        """
        Initializes the AnalysisPlotter.

        Args:
            dataframe (pd.DataFrame): The dataframe containing experiment results.
            x_column (str): Column name to use for the X-axis.
            y_column (str): Column name to use for the Y-axis.
            title (str): Title of the plot.
        """
        self.dataframe = dataframe
        self.x_column = x_column
        self.y_column = y_column
        self.title = title
    
    def _detect_stabilization(self, x_values: np.ndarray, y_values: np.ndarray, patience: int, min_delta: float):
        """
        Detects stabilization point where the Y values stop decreasing significantly.
        """
        best_value = y_values[0]
        wait = 0

        for i in range(1, len(y_values)):
            if y_values[i] < best_value - min_delta:
                best_value = y_values[i]
                wait = 0  # Reset patience
            else:
                wait += 1
                if wait >= patience:
                    return x_values[i]
        
        return None
    
    def plot_data(self, architectures: list, pruning_distributions: list, batch_sizes: list, patience: int, min_delta: float):
        """
        Plots data for specified architectures, pruning distributions, and batch sizes.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        
        palette = sns.color_palette("Set2")
        
        for arch in architectures:
            for num, pd in enumerate(pruning_distributions):
                for batch in batch_sizes:
                    filtered = self.dataframe[(
                        (self.dataframe["Architecture"] == arch) &
                        (self.dataframe["Pruning Distribution"] == pd) &
                        (self.dataframe["BATCH_SIZE"] == batch)
                    ) | (
                        (self.dataframe["Architecture"] == arch) &
                        (self.dataframe["BATCH_SIZE"] == batch) &
                        (self.dataframe[self.x_column] == 0)
                    )]
                    
                    if filtered.empty:
                        continue
                    
                    filtered = filtered.sort_values(by=self.x_column)
                    
                    x_values = filtered[self.x_column].values
                    y_values = filtered[self.y_column].values
                    
                    std_col = "STD " + self.y_column.replace("Mean ", "")
                    if std_col in filtered.columns:
                        std_values = filtered[std_col].values
                    else:
                        std_values = np.zeros_like(y_values)
                    
                    label = f"{arch}-{pd}-Batch {batch}"
                    sns.lineplot(x=x_values, y=y_values, marker='o', label=label, errorbar=None)
                    
                    if np.any(std_values > 0):
                        plt.fill_between(x_values, y_values - std_values, y_values + std_values, alpha=0.2)
                    
                    stabilization_x = self._detect_stabilization(x_values, y_values, patience, min_delta)
                    if stabilization_x is not None:
                        plt.axvline(x=stabilization_x, linestyle='--', linewidth=0.8, color='gray', label=f'Stabilization {label}')
                    
                    # Add unpruned model label at GPR=0
                    if 0 in x_values and num == 0:
                        unpruned_value = y_values[np.where(x_values == 0)][0]
                        plt.scatter(0, unpruned_value, color='red', s=100, zorder=3, label=f"Unpruned Model ({arch}-Batch {batch})")
        
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.title(self.title)
        plt.legend()
        plt.show()
