import logging
from importlib.metadata import PackageNotFoundError, version

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("coin-sentiment")
except PackageNotFoundError:
    __version__ = "0.0.0"
