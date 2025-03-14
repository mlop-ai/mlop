import logging
import os
from datetime import datetime

from . import sets
from .log import setup_logger
from .ops import Ops
from .sets import Settings
from .sys import System
from .util import gen_id, to_json

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Init"


class OpsInit:
    def __init__(self, config) -> None:
        self.kwargs = None
        self.config: dict[str, any] = config

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
        self.settings.meta = []  # TODO: find a better way to de-reference meta

        if self.settings.mode == "noop":
            self.settings.disable_iface = True
            self.settings.disable_store = True
        else:
            os.makedirs(f"{setup_settings.work_dir()}/files", exist_ok=True)
            global logger
            setup_logger(
                self.settings, logger, console=logging.getLogger("console"), op=True
            )

            self.settings.system = System(self.settings)
            to_json([self.settings.system.info()], f"{settings.work_dir()}/sys.json")


def init(
    dir: str | None = None,
    project: str | None = None,
    name: str | None = None,
    id: str | None = None,
    config: dict | str | None = None,
    settings: Settings | dict[str, any] | None = {},
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
        op = OpsInit(config=config)
        op.setup(settings=settings)
        return op.init()
    except Exception as e:
        logger.critical("%s: failed, %s", tag, e)  # add early logger
        raise e
