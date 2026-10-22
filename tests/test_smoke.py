"""Testes mínimos da fundação do projeto."""

from importlib import import_module, util


def test_src_package_exposes_version() -> None:
    """A versão pública permite identificar a fundação instalada."""
    assert util.find_spec("src") is not None
    src = import_module("src")
    assert isinstance(src.__version__, str)
    assert src.__version__ == "0.1.0"
