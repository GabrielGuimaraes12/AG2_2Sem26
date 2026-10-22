from collections.abc import Callable

import pandas as pd
import pytest


@pytest.fixture
def valid_raw_frame() -> Callable[[], pd.DataFrame]:
    """Cria dados brutos mínimos com as duas classes e regiões conhecidas."""

    def factory() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Channel": ["Retail", "HoReCa", "Retail", "HoReCa"],
                "Region": ["Other", "Lisbon", "Oporto", "Other"],
                "Fresh": [12669, 7057, 6353, 13265],
                "Milk": [9656, 9810, 8808, 1196],
                "Grocery": [7561, 9568, 7684, 4221],
                "Frozen": [214, 1762, 2405, 6404],
                "Detergents_Paper": [2674, 3293, 3516, 507],
                "Delicatessen": [1338, 1776, 7844, 1788],
            }
        )

    return factory
