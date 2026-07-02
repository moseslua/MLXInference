from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model-id",
        action="store",
        default="fake://deterministic",
        help="Model id for optional MLX-LM parity checks.",
    )


@pytest.fixture
def model_id(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--model-id"))
