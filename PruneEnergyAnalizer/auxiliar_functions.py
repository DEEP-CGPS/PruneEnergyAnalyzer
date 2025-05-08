import pandas as pd

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
    Adds a compression ratio column (%) for each model, comparing it to its corresponding unpruned version
    (with the same architecture and batch size), based on a specified metric.

    The compression ratio is calculated as:
        Compression Ratio (%) = 100 - (pruned_value / unpruned_value) * 100

    Args:
        df (pd.DataFrame): A DataFrame containing experiment results. Must include:
                           'MODEL_NAME', 'ARCHITECTURE', 'BATCH_SIZE', and the given metric.
        unpruned_names (list): List of unpruned model filenames from the 'MODEL_NAME' column.
        metric (str): Metric to use for compression ratio. Must be either 'Parameters' or 'FLOPs'.
        decimals (int): Number of decimal places to round the result.

    Returns:
        pd.DataFrame: A copy of the original DataFrame with one new column:
                      'Compression Ratio (metric) [%]'.

    Raises:
        ValueError: If an invalid metric is provided.
    """
    if metric not in ["Parameters", "FLOPs"]:
        raise ValueError("Metric must be either 'Parameters' or 'FLOPs'.")

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

