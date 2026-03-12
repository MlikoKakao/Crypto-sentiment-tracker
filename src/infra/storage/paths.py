from pathlib import Path


def get_demo_data_path(demo_path: str) -> Path:
    base = DEMO_DIR if DEMO_MODE else DATA_DIR
    return base.joinpath(demo_path)
