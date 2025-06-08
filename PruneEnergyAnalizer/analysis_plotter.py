import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
from typing import List

class AnalysisPlotter:
    """
    Class for analyzing and visualizing the impact of pruning on energy consumption and performance metrics.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing results from pruning experiments, with columns for architecture, pruning distribution, batch size, metrics, etc.
    x_column : str
        Name of the column to use for the x-axis (e.g., 'GPR', 'BATCH_SIZE').
    y_column : str
        Name of the column to use for the y-axis (e.g., 'Mean Energy per Sample (J)', 'FPS').
    title : str
        Title to use for all generated plots.
    """
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        x_column: str, 
        y_column: str, 
        title: str
    ):
        self.dataframe = dataframe
        self.x_column = x_column
        self.y_column = y_column
        self.title = title

    def plot_data(
        self, 
        architectures: List[str], 
        pruning_distributions: List[str], 
        batch_sizes: List[int]
    ) -> None:
        """
        Generate and display multiple plots to visualize how pruning and model choices affect the selected metric.

        Parameters
        ----------
        architectures : List[str]
            List of model architecture names to include in the plots (e.g., ['AlexNet', 'VGG11']).
        pruning_distributions : List[str]
            List of pruning distribution names to include (e.g., ['random_PD1', 'random_PD2']).
        batch_sizes : List[int]
            List of batch sizes to plot (e.g., [1, 32, 128]).

        Notes
        -----
        - The method generates three types of plots:
            1. Original line plot with error bands.
            2. Smoothed curve using spline interpolation.
            3. Moving average curve.
        - Make sure your DataFrame has the necessary columns (see library documentation for details).
        """
        sns.set(style="whitegrid")

        def get_combined_filtered(arch: str, pdistr: str, batch: int) -> pd.DataFrame:
            """
            Filter the DataFrame for a given architecture, pruning distribution, and batch size.

            Returns a DataFrame sorted by the x_column.
            """
            filtered = self.dataframe[
                ((self.dataframe["Architecture"] == arch) &
                 (self.dataframe["Pruning Distribution"] == pdistr) &
                 (self.dataframe["BATCH_SIZE"] == batch)) |
                ((self.dataframe["Architecture"] == arch) &
                 (self.dataframe["BATCH_SIZE"] == batch) &
                 (self.dataframe[self.x_column] == 0))
            ]
            return filtered.sort_values(by=self.x_column)

        # Plot 1: Original
        plt.figure(figsize=(8, 5))
        for arch in architectures:
            for num, pdistr in enumerate(pruning_distributions):
                for batch in batch_sizes:
                    filtered = get_combined_filtered(arch, pdistr, batch)
                    if filtered.empty:
                        continue
                    x_values = filtered[self.x_column].astype('int64').values
                    y_values = filtered[self.y_column].values
                    std_col = "STD " + self.y_column.replace("Mean ", "")
                    std_values = filtered[std_col].values if std_col in filtered.columns else np.zeros_like(y_values)
                    label = f"{arch}-{pdistr}-Batch {batch}"
                    sns.lineplot(x=x_values, y=y_values, marker='o', label=label, errorbar=None)
                    if np.any(std_values > 0):
                        plt.fill_between(x_values, y_values - std_values, y_values + std_values, alpha=0.2)
                    if 0 in x_values and num == 0:
                        unpruned_value = y_values[np.where(x_values == 0)][0]
                        plt.scatter(0, unpruned_value, color='red', s=100, zorder=3, label=f"Unpruned Model ({arch}-Batch {batch})")
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.title(f"{self.title}")
        plt.legend(fontsize='small')
        plt.show()

        # Plot 2: Smoothed with spline interpolation
        plt.figure(figsize=(8, 5))
        for arch in architectures:
            for num, pdistr in enumerate(pruning_distributions):
                for batch in batch_sizes:
                    filtered = get_combined_filtered(arch, pdistr, batch)
                    if filtered.empty:
                        continue
                    x_values = filtered[self.x_column].values
                    y_values = filtered[self.y_column].values
                    if len(x_values) >= 4:
                        spline = make_interp_spline(x_values, y_values, k=3)
                        x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
                        y_smooth = spline(x_smooth)
                        label = f"{arch}-{pdistr}-Batch {batch}"
                        plt.plot(x_smooth, y_smooth, label=label)
                        if 0 in x_values and num == 0:
                            unpruned_value = y_values[np.where(x_values == 0)][0]
                            plt.scatter(0, unpruned_value, color='red', s=100, zorder=3, label=f"Unpruned Model ({arch}-Batch {batch})")
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.title(f"Spline Smooth: {self.title}")
        plt.legend(fontsize='small')
        plt.show()

        # Plot 3: Moving average (combined)
        plt.figure(figsize=(8, 5))
        for arch in architectures:
            for batch in batch_sizes:
                for num, pdistr in enumerate(pruning_distributions):
                    filtered = get_combined_filtered(arch, pdistr, batch)
                    if filtered.empty:
                        continue
                    x_values = filtered[self.x_column].values
                    y_values = filtered[self.y_column].values

                    if 0 in x_values:
                        # Ensure GPR=0 uses the exact value of the unpruned model
                        unpruned_value = y_values[np.where(x_values == 0)][0]

                    if len(y_values) >= 3:
                        y_moving_avg = np.convolve(y_values, np.ones(3)/3, mode='same')
                        if 0 in x_values:
                            idx_0 = np.where(x_values == 0)[0][0]
                            y_moving_avg[idx_0] = unpruned_value  # Force exact value at GPR=0

                        label = f"{arch}-{pdistr}-Batch {batch}"
                        plt.plot(x_values, y_moving_avg, label=label)
                        if 0 in x_values and num == 0:
                            plt.scatter(0, unpruned_value, color='red', s=100, zorder=3, label=f"Unpruned Model ({arch}-Batch {batch})")
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.title(f"Moving Average: {self.title}")
        plt.legend(fontsize='small')
        plt.show()


def plot_energy_and_metric_curve(
    dataframe: pd.DataFrame,
    x_column: str,
    energy_column: str,
    metric_column: str,
    title: str = "Energy vs Metric Trade-off"
):
    """
    Plots energy consumption and a secondary model metric over a specified X-axis.

    Args:
        dataframe (pd.DataFrame): DataFrame with experiment data.
        x_column (str): Column name for the X-axis (e.g., 'GPR', 'Compression Ratio').
        energy_column (str): Column name for the energy values (e.g., 'Mean Energy per Sample (J)').
        metric_column (str): Column name for the model metric (e.g., 'Accuracy (%)').
        title (str): Title of the plot.
    """
    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot energy (left Y-axis)
    sns.lineplot(
        data=dataframe,
        x=x_column,
        y=energy_column,
        marker='o',
        color='tab:blue',
        ax=ax1
    )
    ax1.set_ylabel(energy_column, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot metric (right Y-axis)
    ax2 = ax1.twinx()
    sns.lineplot(
        data=dataframe,
        x=x_column,
        y=metric_column,
        marker='s',
        color='tab:green',
        ax=ax2
    )
    ax2.set_ylabel(metric_column, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Add unpruned model point (GPR = 0)
    unpruned = dataframe[dataframe[x_column] == 0]
    handles = []

    if not unpruned.empty:
        unpruned_energy = unpruned[energy_column].values[0]
        unpruned_metric = unpruned[metric_column].values[0]

        h1 = ax1.scatter(0, unpruned_energy, color='red', s=100, zorder=5)
        h2 = ax2.scatter(0, unpruned_metric, color='red', s=100, zorder=5)

        # Build legend handles manually
        handles = [
            plt.Line2D([0], [0], color='tab:blue', marker='o', label=energy_column),
            plt.Line2D([0], [0], color='tab:green', marker='s', label=metric_column),
            plt.Line2D([0], [0], color='red', marker='o', linestyle='', markersize=10, label='Unpruned Model')
        ]
    else:
        handles = [
            plt.Line2D([0], [0], color='tab:blue', marker='o', label=energy_column),
            plt.Line2D([0], [0], color='tab:green', marker='s', label=metric_column)
        ]

    # Plot layout
    ax1.set_xlabel(x_column)
    fig.suptitle(title)
    ax1.legend(handles=handles, loc='upper left', fontsize='small')
    fig.tight_layout()
    plt.show()