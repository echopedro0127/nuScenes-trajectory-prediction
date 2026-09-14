def test_package_exposes_version() -> None:
    from trajectory_prediction import __version__

    assert __version__ == "0.1.0"
