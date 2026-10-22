"""Ponto de entrada da classificação interativa."""

from pathlib import Path

from src.cli import InputCancelled, InputFunction, OutputFunction, collect_sample
from src.data import DataValidationError, load_dataset, prepare_dataset, split_features_target
from src.model import (
    ModelTrainingError,
    evaluate_classifier,
    predict_channel,
    split_dataset,
    train_classifier,
)


def default_dataset_path() -> Path:
    """Resolve o CSV pela localização do projeto, sem depender do diretório atual."""
    return Path(__file__).resolve().parent / "data" / "wholesale.csv"


def main(
    input_fn: InputFunction = input, output_fn: OutputFunction = print
) -> int:
    """Treina, avalia e usa o classificador em uma nova amostra."""
    try:
        output_fn("=== Classificador de canal de vendas ===")
        raw = load_dataset(default_dataset_path())
        features, target = split_features_target(prepare_dataset(raw))
        split = split_dataset(features, target)
        model = train_classifier(split.X_train, split.y_train)
        evaluation = evaluate_classifier(model, split.X_test, split.y_test)

        output_fn("\n=== Avaliação no conjunto de teste ===")
        output_fn(evaluation.report)
        output_fn(f"Acurácia: {evaluation.accuracy:.4f}")
        output_fn("\n=== Nova amostra ===")
        sample = collect_sample(input_fn, output_fn)
        output_fn(f"Canal previsto: {predict_channel(model, sample)}")
        return 0
    except (DataValidationError, ModelTrainingError, InputCancelled) as error:
        output_fn(f"Não foi possível concluir: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
