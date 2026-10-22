from pathlib import Path

import pytest

import main as app
from main import default_dataset_path, main


def test_default_dataset_path_is_relative_to_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    path = default_dataset_path()

    assert path.name == "wholesale.csv"
    assert path.parent.name == "data"
    assert path.is_file()


def test_main_trains_reports_and_predicts() -> None:
    answers = iter(["2", "12669", "9656", "7561", "214", "2674", "1338"])
    output: list[str] = []

    exit_code = main(lambda _: next(answers), output.append)

    rendered = "\n".join(output)
    assert exit_code == 0
    assert "HoReCa" in rendered
    assert "Retail" in rendered
    assert "Acurácia" in rendered
    assert "Canal previsto:" in rendered


def test_main_reports_invalid_dataset_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invalid = tmp_path / "empty.csv"
    invalid.write_bytes(b"")
    output: list[str] = []
    monkeypatch.setattr(app, "default_dataset_path", lambda: invalid)

    exit_code = app.main(lambda _: "", output.append)

    assert exit_code == 1
    assert "Não foi possível concluir" in "\n".join(output)
