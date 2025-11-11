import logging
import os
import re


class NoColorFormatter(logging.Formatter):
    ansi_escape = re.compile(
        r"\x1B[@-_][0-?]*[ -/]*[@-~]"
    )  # Regex to match ANSI escape codes

    def format(self, record):
        original = super().format(record)
        return self.ansi_escape.sub("", original)  # Remove ANSI codes


def setup_logger(filename, verbose, logger_name=None):
    # Create the logs directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # If no logger_name is provided, use the module name.
    name = logger_name or __name__
    # Create a logger object
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # If the logger already has handlers, clear them
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = NoColorFormatter("%(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if verbose:
        # Create a stream handler to print logs to console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger
