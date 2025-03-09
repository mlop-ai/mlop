import builtins
import logging
import sys
import time

from .api import make_compat_message_v1
from .util import ANSI

builtins_input = builtins.input
logger = logging.getLogger(f"{__name__.split('.')[0]}")


class ColorFormatter(logging.Formatter):
    def format(self, record):
        prefix = ANSI.bold + ANSI.blue + f"{__name__.split('.')[0]}:" + ANSI.reset
        colors = {
            "DEBUG": ANSI.green,
            "INFO": ANSI.blue,
            "WARNING": ANSI.yellow,
            "ERROR": ANSI.red,
            "CRITICAL": ANSI.purple,
        }
        styles = {
            "DEBUG": " 💬 ",
            "INFO": " 🚀 ",
            "WARNING": " 🚨 ",
            "ERROR": " ⛔ ",
            "CRITICAL": " 🚫 ",
        }
        color = colors.get(record.levelname, "")
        style = styles.get(record.levelname, "")
        # record.msg = f"{color}{record.msg}{ANSI.reset}"
        return f"{prefix}{color}{style}{super().format(record)}{ANSI.reset}"


class ConsoleHandler:
    def __init__(
        self, logger, queue, level=logging.INFO, stream=sys.stdout, type="stdout"
    ):
        self.logger = logger
        self.queue = queue
        self.level = level
        self.stream = stream
        self.type = type
        self.count = 0

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.count += 1
            m = line.rstrip()
            self.queue.put(
                make_compat_message_v1(self.level, m, int(time.time()), self.count)
            )
            self.logger.log(self.level, m)
        self.stream.write(buf)
        self.stream.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def stream_formatter(settings):
    if settings.x_log_level <= logging.DEBUG:
        return ColorFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s %(threadName)-10s:%(process)d "
            "[%(filename)s:%(funcName)s():%(lineno)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        return ColorFormatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )


def input_hook(prompt="", logger=None):
    content = builtins_input(prompt)
    logger.warn(f"{prompt}{content}")
    return content


def setup_logger(settings) -> None:
    global logger
    if len(logger.handlers) != 0:
        return
    logger.setLevel(settings.x_log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter(settings))
    logger.addHandler(stream_handler)

    if not settings.disable_logger:
        setup_logger_file(settings)


def setup_logger_file(settings) -> None:
    file_handler = logging.FileHandler(f"{settings.work_dir()}/{settings.tag}.log")
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(threadName)-10s:%(process)d "
        "[%(filename)s:%(funcName)s():%(lineno)s] %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console = logging.getLogger("console")
    console.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(f"{settings.work_dir()}/sys.log")
    file_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    console.addHandler(file_handler)  # TODO: fix slow file writes
    sys.stdout = ConsoleHandler(
        console, settings.message, logging.INFO, sys.stdout, "stdout"
    )
    sys.stderr = ConsoleHandler(
        console, settings.message, logging.ERROR, sys.stderr, "stderr"
    )

    if settings.mode == "debug":
        builtins.input = lambda prompt="": input_hook(prompt, logger=console)
