import logging
from typing import Any, Dict, Union

import mlop

from .op import Op
from .sets import Settings, setup
from .util import gen_id, get_char

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Init"


class OpInit:
    def __init__(self, config) -> None:
        self.kwargs = None
        self.config: Dict[str, Any] = config

    def init(self) -> Op:
        op = Op(config=self.config, settings=self.settings)
        op.start()
        return op

    def setup(self, settings) -> None:
        self.settings = settings


def init(
    dir: Union[str, None] = None,
    project: Union[str, None] = None,
    name: Union[str, None] = None,
    # id: str | None = None,
    config: Union[dict, str, None] = None,
    settings: Union[Settings, Dict[str, Any], None] = dict(),
    **kwargs,
) -> Op:
    # TODO: remove legacy compat
    dir = kwargs.get("save_dir", dir)

    settings = setup(settings)
    settings.dir = dir if dir else settings.dir
    settings.project = get_char(project) if project else settings.project
    settings._op_name = (
        get_char(name) if name else gen_id(seed=settings.project)
    )  # datetime.now().strftime("%Y%m%d"), str(int(time.time()))
    # settings._op_id = id if id else gen_id(seed=settings.project)

    try:
        op = OpInit(config=config)
        op.setup(settings=settings)
        op = op.init()
        return op
    except Exception as e:
        logger.critical("%s: failed, %s", tag, e)  # add early logger
        raise e


def finish(op: Union[Op, None] = None) -> None:
    if op:
        op.finish()
    else:
        for op in mlop.ops:
            op.finish()
