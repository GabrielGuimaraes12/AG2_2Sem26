from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    FEATURE_COLUMNS,
    MAX_SPENDING,
    PREPARED_COLUMNS,
    DataValidationError,
    load_dataset,
    prepare_dataset,
    split_features_target,
)


def test_prepare_dataset_maps_categories_and_orders_columns(
    valid_raw_frame: Callable[[], pd.DataFrame],
) -> None:
    prepared = prepare_dataset(valid_raw_frame())

    assert list(prepared.columns) == list(PREPARED_COLUMNS)
    assert list(prepared["Channel"]) == [1, 0, 1, 0]
    assert list(prepared["Region"]) == [2, 0, 1, 2]
    assert list(prepared.columns[:-1]) == list(FEATURE_COLUMNS)
    assert all(dtype.kind in "iu" for dtype in prepared.dtypes)


def test_prepare_dataset_rejects_wrong_columns(
    valid_raw_frame: Callable[[], pd.DataFrame],
) -> None:
    raw = valid_raw_frame().drop(columns="Milk")

    with pytest.raises(DataValidationError, match="Milk"):
        prepare_dataset(raw)


@pytest.mark.parametrize(
    ("column", "value"), [("Channel", "Online"), ("Region", "Unknown")]
)
def test_prepare_dataset_rejects_unknown_categories(
    valid_raw_frame: Callable[[], pd.DataFrame], column: str, value: str
) -> None:
    raw = valid_raw_frame()
    raw.loc[0, column] = value

    with pytest.raises(DataValidationError, match=value):
        prepare_dataset(raw)


def test_prepare_dataset_rejects_negative_spending(
    valid_raw_frame: Callable[[], pd.DataFrame],
) -> None:
    raw = valid_raw_frame()
    raw.loc[0, "Fresh"] = -1

    with pytest.raises(DataValidationError, match="Fresh"):
        prepare_dataset(raw)


def test_prepare_dataset_rejects_spending_above_int64(
    valid_raw_frame: Callable[[], pd.DataFrame],
) -> None:
    raw = valid_raw_frame()
    raw["Fresh"] = pd.Series(
        [MAX_SPENDING + 1, 7057, 6353, 13265], dtype="uint64"
    )

    with pytest.raises(DataValidationError, match="Fresh.*máximo|máximo.*Fresh"):
        prepare_dataset(raw)


def test_prepare_dataset_rejects_missing_values(
    valid_raw_frame: Callable[[], pd.DataFrame],
) -> None:
    raw = valid_raw_frame()
    raw.loc[0, "Milk"] = None

    with pytest.raises(DataValidationError, match="Milk.*ausente|ausente.*Milk"):
        prepare_dataset(raw)


@pytest.mark.parametrize("value", [1.5, "dez"])
def test_prepare_dataset_rejects_non_integer_spending(
    valid_raw_frame: Callable[[], pd.DataFrame], value: float | str
) -> None:
    raw = valid_raw_frame()
    raw["Grocery"] = raw["Grocery"].astype(object)
    raw.loc[0, "Grocery"] = value

    with pytest.raises(DataValidationError, match="Grocery"):
        prepare_dataset(raw)


def test_real_dataset_has_expected_shape_and_schema() -> None:
    raw = load_dataset(Path(__file__).parents[1] / "data" / "wholesale.csv")
    prepared = prepare_dataset(raw)
    features, target = split_features_target(prepared)

    assert raw.shape == (440, 8)
    assert list(prepared.columns) == list(PREPARED_COLUMNS)
    assert not prepared.isna().any().any()
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert set(target.unique()) == {0, 1}


@pytest.mark.parametrize("content", [b"", b"\xff\xfe\xfa"])
def test_load_dataset_wraps_invalid_files(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "invalid.csv"
    path.write_bytes(content)

    with pytest.raises(DataValidationError, match="invalid.csv"):
        load_dataset(path)
