from __future__ import annotations

import sys

import pytest
import os

os.environ["CI"] = "1"  # Indicate that we are in a CI environment


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--fast", action="store_true", default=False)  # , nargs="+", default=[0], type=int)
    parser.addoption("--mp-only", action="store_true", default=False)


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]):
    for item in items:
        marker = item.get_closest_marker("length")
        if marker is None:
            item.add_marker(pytest.mark.length(speed="fast"))


@pytest.fixture(autouse=True)
def clean_sys_argv(monkeypatch):
    """
    Automatically reset sys.argv for each test to avoid pytest CLI arguments interfering
    with custom argument parsers in tests.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        None
    """
    monkeypatch.setattr(sys, "argv", ["pytest"])
