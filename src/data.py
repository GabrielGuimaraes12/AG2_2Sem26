"""Leitura e preparação do conjunto Wholesale customers."""

from pathlib import Path

import pandas as pd

RAW_COLUMNS = (
    "Channel",
    "Region",
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicatessen",
)
FEATURE_COLUMNS = (
    "Region",
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents Paper",
    "Delicatessen",
)
PREPARED_COLUMNS = (*FEATURE_COLUMNS, "Channel")
SPENDING_COLUMNS = RAW_COLUMNS[2:]
CHANNEL_MAP = {"HoReCa": 0, "Retail": 1}
REGION_MAP = {"Lisbon": 0, "Oporto": 1, "Other": 2}
MAX_SPENDING = 2**63 - 1


class DataValidationError(ValueError):
    """Indica que o arquivo não respeita o contrato do conjunto."""


def load_dataset(path: Path) -> pd.DataFrame:
    """Carrega um CSV preservando seus valores brutos."""
    try:
        return pd.read_csv(path)
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise DataValidationError(f"Não foi possível ler o arquivo {path}: {error}") from error


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida, codifica categorias e entrega as colunas na ordem do modelo."""
    missing_columns = [column for column in RAW_COLUMNS if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in RAW_COLUMNS]
    if missing_columns or extra_columns:
        details = []
        if missing_columns:
            details.append(f"ausentes: {', '.join(missing_columns)}")
        if extra_columns:
            details.append(f"inesperadas: {', '.join(extra_columns)}")
        raise DataValidationError(f"Colunas inválidas ({'; '.join(details)}).")

    columns_with_nulls = [column for column in RAW_COLUMNS if frame[column].isna().any()]
    if columns_with_nulls:
        raise DataValidationError(
            f"Valores ausentes nas colunas: {', '.join(columns_with_nulls)}."
        )

    for column, allowed in (("Channel", CHANNEL_MAP), ("Region", REGION_MAP)):
        unknown = sorted(set(frame[column]) - set(allowed))
        if unknown:
            values = ", ".join(str(value) for value in unknown)
            raise DataValidationError(f"Categoria desconhecida em {column}: {values}.")

    normalized_spending: dict[str, pd.Series] = {}
    for column in SPENDING_COLUMNS:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise DataValidationError(f"A coluna {column} deve conter somente inteiros.")
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any() or ((numeric % 1) != 0).any():
            raise DataValidationError(f"A coluna {column} deve conter somente inteiros.")
        if (numeric < 0).any():
            raise DataValidationError(f"A coluna {column} não aceita gastos negativos.")
        if (numeric > MAX_SPENDING).any():
            raise DataValidationError(
                f"A coluna {column} excede o máximo suportado de {MAX_SPENDING}."
            )
        normalized_spending[column] = numeric.astype("int64")

    prepared = frame.copy()
    for column, series in normalized_spending.items():
        prepared[column] = series
    prepared["Channel"] = prepared["Channel"].map(CHANNEL_MAP)
    prepared["Region"] = prepared["Region"].map(REGION_MAP)
    prepared = prepared.rename(columns={"Detergents_Paper": "Detergents Paper"})
    return prepared.loc[:, PREPARED_COLUMNS].astype("int64")


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa recursos e rótulo de um DataFrame já preparado."""
    return frame.loc[:, FEATURE_COLUMNS].copy(), frame["Channel"].copy()
