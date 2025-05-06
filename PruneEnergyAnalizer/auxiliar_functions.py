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
