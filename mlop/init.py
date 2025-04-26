import logging
import os
import time
from datetime import datetime

import mlop

from . import sets
from .op import Op
from .sets import Settings
from .util import gen_id

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Init"


class OpInit:
    def __init__(self, config) -> None:
        self.kwargs = None
        self.config: dict[str, any] = config

    def init(self) -> Op:
        op = Op(config=self.config, settings=self.settings)
        op.start()
        return op

    def setup(self, settings) -> None:
        self.settings = settings
        self.settings.meta = []  # TODO: find a better way to de-reference meta


def init(
    dir: str | None = None,
    project: str | None = None,
    name: str | None = None,
    # id: str | None = None,
    config: dict | str | None = None,
    settings: Settings | dict[str, any] | None = {},
) -> Op:
    if not isinstance(settings, Settings):  # isinstance(settings, dict)
        default = Settings()
        default.update(settings)
        settings = default

    settings.dir = dir if dir else settings.dir
    settings.project = project if project else settings.project

    settings._op_name = (
        name if name else gen_id(seed=settings.project)
    )  # datetime.now().strftime("%Y%m%d"), str(int(time.time()))
    # settings._op_id = id if id else gen_id(seed=settings.project)

    try:
        op = OpInit(config=config)
        op.setup(settings=settings)
        op = op.init()

        # set globals
        if mlop.ops is None:
            mlop.ops = []
        mlop.ops.append(op)
        mlop.log, mlop.alert, mlop.watch = op.log, op.alert, op.watch
        return op
    except Exception as e:
        logger.critical("%s: failed, %s", tag, e)  # add early logger
        raise e


def finish(op: Op = None) -> None:
    if op:
        op.finish()
    else:
        for op in mlop.ops:
            op.finish()
