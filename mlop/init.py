import builtins
import logging
import os
import sys
from datetime import datetime

from . import sets
from .log import ColorFormatter, ConsoleHandler, input_hook
from .ops import Ops
from .sets import Settings
from .sys import System
from .util import gen_id, to_json

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Init"


class OpsInit:
    def __init__(self) -> None:
        self.kwargs = None
        self.config: dict[str, any] = {}

    def init(self) -> Ops:
        op = Ops(config=self.config, settings=self.settings)
        op.start()
        return op

    def setup(self, settings) -> None:
        init_settings = Settings()
        setup_settings = sets.setup(settings=init_settings).settings

        # TODO: handle login and settings validation here
        setup_settings.update(settings)
        self.settings = setup_settings

        if self.settings.mode == "noop":
            self.settings.disable_iface = True
            self.settings.disable_store = True
        else:
            os.makedirs(f"{setup_settings.work_dir()}/files", exist_ok=True)
            self._logger_setup()

    def _logger_setup(self) -> None:
        global logger
        logger.setLevel(self.settings.x_log_level)

        file_handler = logging.FileHandler(
            f"{self.settings.work_dir()}/{self.settings.tag}.log"
        )
        file_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(threadName)-10s:%(process)d "
            "[%(filename)s:%(funcName)s():%(lineno)s] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        if self.settings.x_log_level <= logging.DEBUG:
            stream_formatter = ColorFormatter(
                "%(asctime)s.%(msecs)03d %(levelname)-8s %(threadName)-10s:%(process)d "
                "[%(filename)s:%(funcName)s():%(lineno)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            stream_formatter = ColorFormatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S",
            )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        console = logging.getLogger("console")
        console.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(f"{self.settings.work_dir()}/sys.log")
        file_formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        console.addHandler(file_handler)  # TODO: fix slow file writes
        sys.stdout = ConsoleHandler(console, self.settings.message, logging.INFO, sys.stdout, "stdout")
        sys.stderr = ConsoleHandler(console, self.settings.message, logging.ERROR, sys.stderr, "stderr")

        if self.settings.mode == "debug":
            builtins.input = lambda prompt="": input_hook(prompt, logger=console)
        
        self.settings.system = System(self.settings)
        to_json([self.settings.system.info()], f"{self.settings.work_dir()}/sys.json")


def init(
    dir: str | None = None,
    project: str | None = None,
    name: str | None = None,
    id: str | None = None,
    config: dict | str | None = None,
    settings: Settings | dict[str, any] | None = None,
) -> Ops:
    if not isinstance(settings, Settings):  # isinstance(settings, dict)
        default = Settings()
        default.update(settings)
        settings = default

    settings.dir = dir if dir else settings.dir
    settings.project = project if project else settings.project

    settings._op_name = name if name else datetime.now().strftime("%Y%m%d")
    settings._op_id = id if id else gen_id()

    try:
        op = OpsInit()
        op.setup(settings=settings)
        return op.init()
    except Exception as e:
        logger.critical("%s: failed, %s", tag, e)  # add early logger
        raise e
