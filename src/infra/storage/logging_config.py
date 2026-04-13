import os
import logging
import sys

class DebugOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.DEBUG

# Seems like way more code than needed for just a simple fork of app.log/debug.log,
# but it's just AI made and not integral to the logic of the app.


def configure_logging() -> None:
    os.makedirs("logs", exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.handlers:
        root_logger.handlers.clear()

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    app_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    debug_handler = logging.FileHandler("logs/debug.log", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    debug_handler.addFilter(DebugOnlyFilter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(app_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(console_handler)
    logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)
