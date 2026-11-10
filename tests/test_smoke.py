from mcs_date import __version__
from mcs_date.__main__ import main


def test_version_is_non_empty_string() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_main_returns_zero() -> None:
    assert main([]) == 0
