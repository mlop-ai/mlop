import logging
import os
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import mlop
from mlop.util import import_lib

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.loggers.utilities import _scan_checkpoints
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
else:
    torch = import_lib("torch")
    ModelCheckpoint = getattr(
        import_lib("lightning.pytorch.callbacks.model_checkpoint"),
        "ModelCheckpoint",
        object,
    )
    Logger = getattr(import_lib("lightning.pytorch.loggers"), "Logger", object)
    rank_zero_only = getattr(
        import_lib("lightning.pytorch.utilities.rank_zero"),
        "rank_zero_only",
        lambda fn: fn,
    )
    _scan_checkpoints = getattr(
        import_lib("lightning.pytorch.loggers.utilities"), "_scan_checkpoints", None
    )

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Lightning"


class MLOPLogger(Logger):
    def __init__(self, op=None, **kwargs):
        super().__init__()
        if hasattr(op, "_log"):
            self.op = op
        elif mlop.ops and len(mlop.ops) > 0:
            self.op = mlop.ops[-1]
        else:
            if "project" not in kwargs:
                kwargs["project"] = "lightning"
            self.op = mlop.init(**kwargs)
        self._checkpoint: Optional[ModelCheckpoint] = None
        self._time = {}

    @property
    def name(self) -> str:
        return getattr(self.op.settings, "_op_name")

    @property
    @rank_zero_only
    def version(self) -> Union[int, str]:
        return getattr(self.op.settings, "_op_id")

    @property
    def experiment(self) -> Any:
        return self.op

    @property
    def root_dir(self) -> Optional[str]:
        return self.op.settings.get_dir()
        # return os.fspath(self.op.settings.get_dir()).parent

    @property
    def save_dir(self) -> Optional[str]:
        return self.op.settings.get_dir()

    @property
    def log_dir(self) -> Optional[str]:
        return self.op.settings.get_dir()

    @property
    def group_separator(self) -> str:
        return "."

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self.op.log(data=metrics, step=step)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        if isinstance(params, Namespace):
            params = vars(params)

        if self.op.config is None:
            self.op.config = {}

        self.op.config.update(params)
        # TODO: trigger an update run

    @rank_zero_only
    def log_file(
        self,
        key: str,
        files: List[Any],
        step: Optional[int] = None,
        ftype: type = mlop.File,
        **kwargs: Any,
    ) -> None:
        for k, v in kwargs.items():
            if not isinstance(v, list) or len(v) != len(files):
                logger.error(
                    f"{tag}: Expected {len(files)} items but found {len(v)} for kwarg '{k}'"
                )
                return

        try:
            d = {
                key: [
                    ftype(item, **kwarg_item)
                    for item, kwarg_item in zip(
                        files,
                        [{k: kwargs[k][i] for k in kwargs} for i in range(len(files))],
                    )
                ]
            }
            self.op.log(data=d, step=step)
        except Exception as e:
            logger.error(
                f"{tag}: Error creating or logging {ftype.__name__} for key '{key}': {e}"
            )

    @rank_zero_only
    def log_image(
        self, key: str, images: List[Any], step: Optional[int] = None, **kwargs: Any
    ) -> None:
        self.log_file(key, images, step, ftype=mlop.Image, **kwargs)

    @rank_zero_only
    def log_audio(
        self, key: str, audios: List[Any], step: Optional[int] = None, **kwargs: Any
    ) -> None:
        self.log_file(key, audios, step, ftype=mlop.Audio, **kwargs)

    @rank_zero_only
    def log_video(
        self, key: str, videos: List[Any], step: Optional[int] = None, **kwargs: Any
    ) -> None:
        self.log_file(key, videos, step, ftype=mlop.Video, **kwargs)

    def save(self) -> None:
        return  # TODO: add save
        self.op._iface.save()

    def finish(self) -> None:
        return  # TODO: add finish
        self.save()
        self.op.finish()

    def log_checkpoint(
        self, checkpoint: "ModelCheckpoint", step: Optional[int] = None
    ) -> None:
        for t, p, s, _ in _scan_checkpoints(checkpoint, self._time):
            self.op.log(
                data={
                    "model": mlop.Artifact(
                        data=p,
                        metadata={
                            "score": s.item() if isinstance(s, torch.Tensor) else s,
                        },
                    )
                },
                step=step,
            )
            self._time[p] = t

    def log_graph(self, model: "torch.nn.Module", **kwargs: Any) -> None:
        self.op.watch(model, **kwargs)

    def watch(self, model: "torch.nn.Module", **kwargs: Any) -> None:
        self.op.watch(model, **kwargs)

    def finalize(self, status: str) -> None:
        if status == "success":
            self.log_checkpoint(self._checkpoint) if self._checkpoint else None
