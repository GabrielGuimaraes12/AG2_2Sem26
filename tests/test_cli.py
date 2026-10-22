from collections.abc import Iterator

import pytest

from src.cli import InputCancelled, collect_sample, prompt_non_negative_int, prompt_region
from src.data import FEATURE_COLUMNS, MAX_SPENDING


def _input_from(values: list[str]) -> tuple[Iterator[str], list[str]]:
    return iter(values), []


@pytest.mark.parametrize("value", ["0", "1", "2"])
def test_prompt_region_accepts_supported_codes(value: str) -> None:
    answers, output = _input_from([value])

    result = prompt_region(lambda _: next(answers), output.append)

    assert result == int(value)
    assert output == []


def test_prompt_region_repeats_after_invalid_value() -> None:
    answers, output = _input_from(["Lisbon", "3", "1"])

    result = prompt_region(lambda _: next(answers), output.append)

    assert result == 1
    assert len(output) == 2


@pytest.mark.parametrize("value", ["0", "42"])
def test_prompt_non_negative_int_accepts_zero_and_positive(value: str) -> None:
    answers, output = _input_from([value])

    result = prompt_non_negative_int("Fresh", lambda _: next(answers), output.append)

    assert result == int(value)
    assert output == []


def test_prompt_non_negative_int_repeats_after_invalid_values() -> None:
    answers, output = _input_from(["", "texto", "1.5", "-1", "8"])

    result = prompt_non_negative_int("Milk", lambda _: next(answers), output.append)

    assert result == 8
    assert len(output) == 4
    assert all("Milk" in message for message in output)


def test_prompt_non_negative_int_repeats_above_supported_maximum() -> None:
    answers, output = _input_from([str(MAX_SPENDING + 1), "8"])

    result = prompt_non_negative_int("Fresh", lambda _: next(answers), output.append)

    assert result == 8
    assert len(output) == 1
    assert "máximo" in output[0]


def test_collect_sample_returns_feature_columns_in_model_order() -> None:
    answers, output = _input_from(["2", "10", "20", "30", "40", "50", "60"])

    sample = collect_sample(lambda _: next(answers), output.append)

    assert list(sample.columns) == list(FEATURE_COLUMNS)
    assert sample.iloc[0].tolist() == [2, 10, 20, 30, 40, 50, 60]
    assert all(dtype.kind in "iu" for dtype in sample.dtypes)


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_collect_sample_handles_cancelled_input(error: BaseException) -> None:
    def cancelled_input(_: str) -> str:
        raise error

    with pytest.raises(InputCancelled, match="cancelada"):
        collect_sample(cancelled_input, lambda _: None)
