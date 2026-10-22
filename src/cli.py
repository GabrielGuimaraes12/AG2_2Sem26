"""Coleta validada de uma amostra pelo terminal."""

from collections.abc import Callable

import pandas as pd

from src.data import FEATURE_COLUMNS, MAX_SPENDING

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class InputCancelled(Exception):
    """Indica que a entrada foi encerrada pelo usuário."""


def _read(prompt: str, input_fn: InputFunction) -> str:
    try:
        return input_fn(prompt)
    except (EOFError, KeyboardInterrupt) as error:
        raise InputCancelled("Entrada cancelada pelo usuário.") from error


def prompt_region(input_fn: InputFunction, output_fn: OutputFunction) -> int:
    """Solicita um dos códigos de região definidos no enunciado."""
    while True:
        value = _read("Região (0 - Lisbon, 1 - Oporto, 2 - Other): ", input_fn).strip()
        if value in {"0", "1", "2"}:
            return int(value)
        output_fn("Região inválida. Digite 0, 1 ou 2.")


def prompt_non_negative_int(
    field: str, input_fn: InputFunction, output_fn: OutputFunction
) -> int:
    """Solicita um inteiro não negativo, repetindo somente o campo inválido."""
    while True:
        value = _read(f"{field}: ", input_fn).strip()
        if value.isdecimal():
            number = int(value)
            if number <= MAX_SPENDING:
                return number
            output_fn(f"{field} deve ser no máximo {MAX_SPENDING}.")
            continue
        output_fn(f"{field} deve ser um número inteiro não negativo.")


def collect_sample(
    input_fn: InputFunction = input, output_fn: OutputFunction = print
) -> pd.DataFrame:
    """Coleta os sete recursos e retorna uma linha na ordem do modelo."""
    values = {
        "Region": prompt_region(input_fn, output_fn),
        "Fresh": prompt_non_negative_int("Fresh", input_fn, output_fn),
        "Milk": prompt_non_negative_int("Milk", input_fn, output_fn),
        "Grocery": prompt_non_negative_int("Grocery", input_fn, output_fn),
        "Frozen": prompt_non_negative_int("Frozen", input_fn, output_fn),
        "Detergents Paper": prompt_non_negative_int(
            "Detergents Paper", input_fn, output_fn
        ),
        "Delicatessen": prompt_non_negative_int("Delicatessen", input_fn, output_fn),
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS, dtype="int64")
