from pathlib import Path



def get_demo_data_path(demo_path: str) -> Path:
    return Path("data/demo/").joinpath(demo_path)
