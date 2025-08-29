import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--fast", action="store_true", default=False)  # , nargs="+", default=[0], type=int)
    parser.addoption("--mp-only", action="store_true", default=False)


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]):
    for item in items:
        # Check if the test has the 'length' marker
        marker = item.get_closest_marker("length")
        if marker is None:
            # If not, add the default marker
            item.add_marker(pytest.mark.length(speed="fast"))
