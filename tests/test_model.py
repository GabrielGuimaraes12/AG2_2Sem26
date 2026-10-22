from pathlib import Path

import pandas as pd
import pytest

from src.data import load_dataset, prepare_dataset, split_features_target
from src.model import (
    ModelTrainingError,
    evaluate_classifier,
    predict_channel,
    split_dataset,
    train_classifier,
)


def _real_features_target() -> tuple[pd.DataFrame, pd.Series]:
    raw = load_dataset(Path(__file__).parents[1] / "data" / "wholesale.csv")
    return split_features_target(prepare_dataset(raw))


def test_split_dataset_uses_reproducible_stratified_80_20_split() -> None:
    features, target = _real_features_target()

    first = split_dataset(features, target)
    second = split_dataset(features, target)

    assert len(first.X_train) == 352
    assert len(first.X_test) == 88
    assert set(first.y_train.unique()) == {0, 1}
    assert set(first.y_test.unique()) == {0, 1}
    assert first.y_train.value_counts().to_dict() == {0: 238, 1: 114}
    assert first.y_test.value_counts().to_dict() == {0: 60, 1: 28}
    assert set(first.X_train.index).isdisjoint(first.X_test.index)
    assert first.X_train.index.equals(second.X_train.index)
    assert first.X_test.index.equals(second.X_test.index)


def test_split_dataset_rejects_class_too_small() -> None:
    features = pd.DataFrame({"value": [1, 2, 3]})
    target = pd.Series([0, 1, 1])

    with pytest.raises(ModelTrainingError, match="classe 0.*duas amostras"):
        split_dataset(features, target)


def test_split_dataset_wraps_impossible_stratified_split() -> None:
    features = pd.DataFrame({"value": [1, 2, 3, 4]})
    target = pd.Series([0, 0, 1, 1])

    with pytest.raises(ModelTrainingError, match="divisão estratificada"):
        split_dataset(features, target)


def test_split_dataset_rejects_empty_dataset() -> None:
    features = pd.DataFrame({"value": pd.Series(dtype="int64")})
    target = pd.Series(dtype="int64")

    with pytest.raises(ModelTrainingError, match="vazio"):
        split_dataset(features, target)


def test_train_classifier_returns_fitted_reproducible_tree() -> None:
    features, target = _real_features_target()
    split = split_dataset(features, target)

    first = train_classifier(split.X_train, split.y_train)
    second = train_classifier(split.X_train, split.y_train)

    assert first.random_state == 42
    assert first.tree_.node_count > 1
    assert list(first.predict(split.X_test)) == list(second.predict(split.X_test))


def test_evaluate_classifier_reports_expected_metrics() -> None:
    features, target = _real_features_target()
    split = split_dataset(features, target)
    model = train_classifier(split.X_train, split.y_train)

    evaluation = evaluate_classifier(model, split.X_test, split.y_test)

    assert 0.0 <= evaluation.accuracy <= 1.0
    for label in ("HoReCa", "Retail", "accuracy", "macro avg", "weighted avg"):
        assert label in evaluation.report


def test_predict_channel_returns_supported_label() -> None:
    features, target = _real_features_target()
    model = train_classifier(features, target)

    prediction = predict_channel(model, features.iloc[[0]])

    assert prediction in {"HoReCa", "Retail"}
