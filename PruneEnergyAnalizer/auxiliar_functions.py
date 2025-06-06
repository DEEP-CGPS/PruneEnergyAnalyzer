import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _parse_model_name_from_string(name: str):
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected model filename format: {name}")
    arch = parts[0]
    if "UNPRUNED" in parts:
        return 0, arch, "UNPRUNED"
    pruning_distribution = next((p for p in parts if "PD" in p), "N/A")
    gpr = next((int(p.split("-")[1]) for p in parts if "GPR" in p), 0)
    return gpr, arch, pruning_distribution

def parse_model_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas GPR, ARCHITECTURE y Pruning Distribution al DataFrame 
    a partir de la columna MODEL_NAME.

    Args:
        df (pd.DataFrame): DataFrame con una columna 'MODEL_NAME'.

    Returns:
        pd.DataFrame: Mismo DataFrame con nuevas columnas agregadas.
    """
    df = df.copy()
    df[["GPR", "Architecture", "Pruning Distribution"]] = df["MODEL_NAME"].apply(
        lambda x: pd.Series(_parse_model_name_from_string(x))
    )
    return df


def add_compression_ratio(
    df: pd.DataFrame, 
    unpruned_names: list, 
    metric: str, 
    decimals: int = 0
) -> pd.DataFrame:
    """
    Adds a compression ratio or energy reduction column (%) for each model,
    comparing it to its corresponding unpruned version (same architecture and batch size),
    based on a specified metric.

    The reduction is calculated as:
        100 - (pruned_value / unpruned_value) * 100

    Args:
        df (pd.DataFrame): DataFrame with model results. Must include:
                           'MODEL_NAME', 'Architecture', 'BATCH_SIZE', and the specified metric.
        unpruned_names (list): List of unpruned model filenames from 'MODEL_NAME'.
        metric (str): Metric to use: 'Parameters', 'FLOPs', or 'Mean Energy per Sample (J)'.
        decimals (int): Number of decimal places to round the result.

    Returns:
        pd.DataFrame: DataFrame with an added percentage reduction column.

    Raises:
        ValueError: If an invalid metric is provided.
    """
    valid_metrics = ["Parameters", "FLOPs", "Mean Energy per Sample (J)"]
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}.")

    # Define column name
    if metric == "Mean Energy per Sample (J)":
        column_name = "% Energy Reduction"
    else:
        column_name = f"Compression Ratio ({metric}) [%]"

    df = df.copy()
    df[column_name] = None

    for name in unpruned_names:
        base_rows = df[df["MODEL_NAME"] == name]
        if base_rows.empty:
            print(f"⚠️ Warning: Unpruned model '{name}' not found in the DataFrame.")
            continue

        for _, base_row in base_rows.iterrows():
            arch = base_row["Architecture"]
            batch_size = base_row["BATCH_SIZE"]
            base_value = base_row[metric]

            mask = (df["Architecture"] == arch) & (df["BATCH_SIZE"] == batch_size)
            df.loc[mask, column_name] = (
                100 - (df.loc[mask, metric] / base_value * 100)
            ).round(decimals)

    return df


def plot_energy_and_metric_curve(
    dataframe: pd.DataFrame,
    architecture: str,
    pruning_distribution: str,
    batch_size: int,
    energy_column: str,
    metric_column: str,
    metric_label: str = "Accuracy (%)",
    title: str = "Energy vs. Accuracy Tradeoff"
):
    """
    Plots energy consumption and an additional model metric over different pruning levels.

    Args:
        dataframe (pd.DataFrame): The data containing GPR, energy, and metric values.
        architecture (str): Model architecture to filter.
        pruning_distribution (str): Pruning distribution to filter.
        batch_size (int): Batch size to filter.
        energy_column (str): Column name for energy values.
        metric_column (str): Column name for additional metric values (e.g., Accuracy).
        metric_label (str): Label for the secondary y-axis.
        title (str): Title of the plot.
    """
    df = dataframe[
        (dataframe["Architecture"] == architecture) &
        (dataframe["Pruning Distribution"] == pruning_distribution) &
        (dataframe["BATCH_SIZE"] == batch_size)
    ].sort_values(by="GPR")

    if df.empty:
        print("No data found for the specified filter.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    sns.set(style="whitegrid")

    # Energy plot
    color1 = "tab:blue"
    ax1.set_xlabel("GPR (%)")
    ax1.set_ylabel(energy_column, color=color1)
    ax1.plot(df["GPR"], df[energy_column], marker='o', color=color1, label=energy_column)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Metric plot
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel(metric_label, color=color2)
    ax2.plot(df["GPR"], df[metric_column], marker='s', linestyle='--', color=color2, label=metric_label)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and layout
    plt.title(title)
    fig.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()
