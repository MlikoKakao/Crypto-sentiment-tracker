__version__ = "1.0.0"

import logging
logging.basicConfig(
    filename="logs/app.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)