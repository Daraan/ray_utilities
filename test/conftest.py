from pytest import Parser


def pytest_addoption(parser: Parser):
    parser.addoption("--fast", action="store_true", default=False)  # , nargs="+", default=[0], type=int)
    parser.addoption("--mp-only", action="store_true", default=False)
