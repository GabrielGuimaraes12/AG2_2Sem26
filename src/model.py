"""Divisão, treino, avaliação e previsão do classificador."""

from dataclasses import dataclass
from typing import cast

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class ModelTrainingError(ValueError):
    """Indica que os dados não permitem treinar o modelo contratado."""


@dataclass(frozen=True)
class DatasetSplit:
    """Partes reproduzíveis usadas para treino e avaliação."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ModelEvaluation:
    """Relatório textual e acurácia calculados no conjunto de teste."""

    report: str
    accuracy: float


def split_dataset(features: pd.DataFrame, target: pd.Series) -> DatasetSplit:
    """Cria divisão estratificada de 80/20 com semente fixa."""
    if target.empty:
        raise ModelTrainingError("O conjunto de dados está vazio.")
    counts = target.value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        label = too_small.index[0]
        raise ModelTrainingError(
            f"A classe {label} precisa de pelo menos duas amostras para estratificação."
        )

    try:
        parts = train_test_split(
            features,
            target,
            test_size=0.20,
            random_state=42,
            stratify=target,
        )
    except ValueError as error:
        raise ModelTrainingError(
            f"Não foi possível criar a divisão estratificada: {error}"
        ) from error
    X_train, X_test, y_train, y_test = parts
    return DatasetSplit(
        X_train=cast(pd.DataFrame, X_train),
        X_test=cast(pd.DataFrame, X_test),
        y_train=cast(pd.Series, y_train),
        y_test=cast(pd.Series, y_test),
    )


def train_classifier(
    features: pd.DataFrame, target: pd.Series
) -> DecisionTreeClassifier:
    """Treina uma árvore de decisão reproduzível."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(features, target)
    return model


def evaluate_classifier(
    model: DecisionTreeClassifier, features: pd.DataFrame, target: pd.Series
) -> ModelEvaluation:
    """Calcula o relatório solicitado e a acurácia do classificador."""
    predictions = model.predict(features)
    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["HoReCa", "Retail"],
        zero_division=0,
    )
    return ModelEvaluation(
        report=cast(str, report),
        accuracy=float(accuracy_score(target, predictions)),
    )


def predict_channel(model: DecisionTreeClassifier, sample: pd.DataFrame) -> str:
    """Converte a previsão numérica do modelo em rótulo público."""
    predicted = int(model.predict(sample)[0])
    labels = {0: "HoReCa", 1: "Retail"}
    if predicted not in labels:
        raise ModelTrainingError(f"O modelo retornou a classe desconhecida {predicted}.")
    return labels[predicted]
