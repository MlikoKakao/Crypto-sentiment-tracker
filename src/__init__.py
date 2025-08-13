import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("crypto-sentiment-tracker")
except Exception:
    __version__ = "0.0.0"
