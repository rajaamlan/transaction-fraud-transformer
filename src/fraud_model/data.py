from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


ID_COLUMNS = {"TransactionID"}


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_ieee_cis(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    transaction = pd.read_csv(data_dir / "train_transaction.csv")
    identity_path = data_dir / "train_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        return transaction.merge(identity, on="TransactionID", how="left")
    return transaction


def infer_feature_columns(
    frame: pd.DataFrame,
    label_column: str,
) -> tuple[list[str], list[str]]:
    candidates = [col for col in frame.columns if col != label_column and col not in ID_COLUMNS]
    categorical: list[str] = []
    numeric: list[str] = []

    for column in candidates:
        if is_object_dtype(frame[column]) or is_string_dtype(frame[column]):
            categorical.append(column)
        else:
            numeric.append(column)

    return numeric, categorical
