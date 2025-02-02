from collections.abc import Mapping
import atexit
import logging
import queue
import threading
import multiprocessing
import time

from .file import File, Image
from .iface import ServerInterface
from .sets import Settings
from .store import DataStore
from .util import dict_to_json

logger = logging.getLogger(f"{__name__.split('.')[0]}")
TAG = "Logging"


class OpsMonitor:
    def __init__(self, op) -> None:
        self.op = op
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self.op._worker, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()  # timeout=self.op.settings.heartbeat_seconds
            self._thread = None


class Ops:
    def __init__(self, config, settings: Settings) -> None:
        self.config = config
        self.settings = settings
        self._monitor = OpsMonitor(op=self)

        self._store = (
            DataStore(settings=settings) if not settings.disable_store else None
        )
        self._iface = (
            ServerInterface(settings=settings) if not settings.disable_iface else None
        )
        self._step = 0
        self._queue = queue.Queue()
        atexit.register(self.finish)

    def start(self) -> None:
        self._monitor.start()
        self._iface.start() if self._iface else None
        logger.info(f"{TAG}: started")

    def log(
        self, data: dict[str, any], step: int | None = None, commit: bool | None = None
    ) -> None:
        """Log run data"""
        if self.settings.mode == "perf":
            self._queue.put((data, step), block=False)
        else:  # bypass queue
            self._log(data=data, step=step)

    def finish(self, exit_code: int | None = None) -> None:
        """Finish logging"""
        self._finish(exit_code=exit_code)
        while not self._queue.empty():
            pass
        self._store.stop() if self._store else None
        self._iface.stop() if self._iface else None  # fixed order
        logger.info(f"{TAG}: finished")

    def _worker(self) -> None:
        while not self._monitor._stop_event.is_set() or not self._queue.empty():
            try:
                self._log(*self._queue.get(block=False))
            except queue.Empty:
                continue # TODO: reduce resource usage with debounce
            except Exception as e:
                logger.critical("%s: failed: %s", TAG, e)
                raise e

    def _log(self, data, step) -> None:
        if not isinstance(data, Mapping):
            e = ValueError(
                f"Data logged must be of dictionary type; received {type(data).__name__} intsead"
            )
            logger.critical("%s: failed: %s", TAG, e)
            raise e
        if any(not isinstance(k, str) for k in data.keys()):
            e = ValueError("Data logged must have keys of string type")
            logger.critical("%s: failed: %s", TAG, e)
            raise e

        t = int(time.time())
        if step is not None:
            if step > self._step:
                self._step = step
        else:
            self._step += 1

        # data = data.copy()  # TODO: check mutability
        d, f = {}, {}
        for k, v in data.items():
            logger.debug(f"{TAG}: added {k} at step {self._step}")
            if isinstance(v, list):
                for e in v:
                    d, f = self._op(d, f, k, e)
            else:
                d, f = self._op(d, f, k, v)

        # d = dict_to_json(d)  # TODO: add serialisation
        self._store.insert(
            data=d, file=f, timestamp=t, step=self._step
        ) if self._store else None
        self._iface.publish(
            data=d, file=f, timestamp=t, step=self._step
        ) if self._iface else None

    def _op(self, d, f, k, v) -> None:
        if isinstance(v, File):
            if isinstance(v, Image):
                v.load(self.settings.work_dir())
            # TODO: add step to serialise data for files
            v._mkcopy(self.settings.work_dir()) # key independent
            # d[k] = int(v._id, 16)
            if k not in f:
                f[k] = [v]
            else:
                f[k].append(v)
        else:
            d[k] = v
        return d, f

    def _finish(self, exit_code) -> None:
        self._monitor.stop()
