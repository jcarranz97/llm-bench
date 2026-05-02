import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="Run tests that require network access (e.g. HuggingFace repo checks)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: mark test as requiring network access")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--network"):
        return
    skip_network = pytest.mark.skip(reason="pass --network to run HuggingFace checks")
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(skip_network)
