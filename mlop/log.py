import builtins
import logging
import sys
import time

from .api import make_compat_message_v1
from .util import ANSI

logger = logging.getLogger(f"{__name__.split('.')[0]}")

bak_input = builtins.input
bak_stdout = sys.stdout
bak_stderr = sys.stderr

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


class ColorFormatter(logging.Formatter):
    def format(self, record):
        prefix = ANSI.bold + ANSI.blue + f"{__name__.split('.')[0]}:" + ANSI.reset
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
            "%(asctime)s | %(message)s",
            # "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )


def input_hook(prompt="", logger=None):
    content = bak_input(prompt)
    logger.warn(f"{prompt}{content}")
    return content


def setup_logger(settings, logger, console=None) -> None:
    if settings._nb_colab():
        rlogger = logging.getLogger()
        for h in rlogger.handlers[:]:  # iter root handlers
            rlogger.removeHandler(h)

    logger.setLevel(settings.x_log_level)

    if len(logger.handlers) == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter(settings))
        logger.addHandler(stream_handler)

    if settings._op_id and not settings.disable_logger:
        if len(console.handlers) > 0:  # full logger
            return
        logger, console = setup_logger_file(settings, logger, console)


def teardown_logger(logger, console=None):
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    if console:
        builtins.input = bak_input
        sys.stdout = bak_stdout  # global bak_stdout
        sys.stderr = bak_stderr
        teardown_logger(logger=console)


def setup_logger_file(settings, logger, console):
    console.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f"{settings.get_dir()}/{settings.tag}.log")
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(threadName)-10s:%(process)d "
        "[%(filename)s:%(funcName)s():%(lineno)s] %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    file_handler = logging.FileHandler(f"{settings.get_dir()}/sys.log")
    file_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    console.addHandler(file_handler)  # TODO: fix slow file writes

    if settings.mode == "debug":
        builtins.input = lambda prompt="": input_hook(prompt, logger=console)
    sys.stdout = ConsoleHandler(
        console, settings.message, logging.INFO, sys.stdout, "stdout"
    )
    sys.stderr = ConsoleHandler(
        console, settings.message, logging.ERROR, sys.stderr, "stderr"
    )

    return logger, console
