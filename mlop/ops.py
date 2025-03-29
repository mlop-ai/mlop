import atexit
import logging
import multiprocessing
import os
import queue
import signal
import threading
import time
from collections.abc import Mapping

from .api import make_compat_start_v1
from .auth import login
from .file import File, Image
from .iface import ServerInterface
from .log import setup_logger, teardown_logger
from .sets import Settings
from .store import DataStore
from .sys import System
from .util import dict_to_json, to_json

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Logging"


class OpsMonitor:
    def __init__(self, op) -> None:
        self.op = op
        self._stop_event = threading.Event()
        self._thread = None
        self._thread_monitor = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.op._worker, args=(self._stop_event.is_set,), daemon=True
            )
            self._thread.start()
        if self._thread_monitor is None:
            self._thread_monitor = threading.Thread(
                target=self._worker_monitor,
                args=(self._stop_event.is_set,),
                daemon=True,
            )
            self._thread_monitor.start()

    def stop(self) -> None:
        self._stop_event.set()
        for t in [self._thread, self._thread_monitor]:
            if t is not None:
                t.join()  # timeout=self.op.settings.x_sys_sampling_interval
                t = None

    def _worker_monitor(self, stop):
        while not stop():
            self.op._iface.publish(
                data=self.op.settings.system.monitor(),
                file=None,
                timestamp=int(time.time()),
                step=self.op._step,
            ) if self.op._iface else None
            time.sleep(self.op.settings.x_sys_sampling_interval)


class Ops:
    def __init__(self, config, settings) -> None:
        self.config = config
        self.settings = settings
        self._monitor = OpsMonitor(op=self)

        if self.settings.mode == "noop":
            self.settings.disable_iface = True
            self.settings.disable_store = True
        else:
            # TODO: set up tmp dir
            login()
            self.settings.system = System(self.settings)
            tmp_iface = ServerInterface(config=config, settings=settings)
            r = tmp_iface._post_v1(
                self.settings.url_start, # create-run
                tmp_iface.headers,
                make_compat_start_v1(self.config, self.settings, self.settings.system.info()),
                client=tmp_iface.client,
            )
            self.settings.url_view = r.json()["url"]
            self.settings._op_id = r.json()['runId']
            logger.info(f"{tag}: started run {str(self.settings._op_id)}")

            os.makedirs(f"{self.settings.get_dir()}/files", exist_ok=True)
            setup_logger(
                settings=self.settings, logger=logger, console=logging.getLogger("stdout")
            ) # global logger
            to_json([self.settings.system.info()], f"{self.settings.get_dir()}/sys.json")

        self._store = (
            DataStore(config=config, settings=settings)
            if not settings.disable_store
            else None
        )
        self._iface = (
            ServerInterface(config=config, settings=settings)
            if not settings.disable_iface
            else None
        )
        self._step = 0
        self._queue = queue.Queue()
        atexit.register(self.finish)

    def start(self) -> None:
        self._iface.start() if self._iface else None
        self._monitor.start()
        logger.debug(f"{tag}: started")

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
        try: 
            self._finish(exit_code=exit_code)
            while not self._queue.empty():
                time.sleep(self.settings.x_internal_check_process)
            self.settings._op_status = 0
            self._store.stop() if self._store else None
            self._iface.stop() if self._iface else None  # fixed order
        except KeyboardInterrupt as e:
            self.settings._op_status = signal.SIGINT.value
            self._iface._update_status(self.settings)
            logger.critical("%s: interrupted %s", tag, e)
        logger.debug(f"{tag}: finished")
        teardown_logger(logger, console=logging.getLogger("stdout"))

    def _worker(self, stop) -> None:
        while not stop() or not self._queue.empty():
            try:
                # if queue seems empty, wait for x_internal_check_process before it considers it empty to save compute
                self._log(
                    *self._queue.get(
                        block=True, timeout=self.settings.x_internal_check_process
                    )
                )
            except queue.Empty:
                continue
            except Exception as e:
                time.sleep(self.settings.x_internal_check_process)  # debounce
                logger.critical("%s: failed: %s", tag, e)
                raise e

    def _log(self, data, step) -> None:
        if not isinstance(data, Mapping):
            e = ValueError(
                f"Data logged must be of dictionary type; received {type(data).__name__} intsead"
            )
            logger.critical("%s: failed: %s", tag, e)
            raise e
        if any(not isinstance(k, str) for k in data.keys()):
            e = ValueError("Data logged must have keys of string type")
            logger.critical("%s: failed: %s", tag, e)
            raise e

        t = int(time.time())
        if step is not None:
            if step > self._step:
                self._step = step
        else:
            self._step += 1

        # data = data.copy()  # TODO: check mutability
        d, f, m = {}, {}, []
        for k, v in data.items():
            if k not in self.settings.meta:
                m.append(k)
                self.settings.meta.append(k)
                # d[f"{self.settings.x_meta_label}{k}"] = 0
                logger.debug(f"{tag}: added {k} at step {self._step}")
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
        self._iface._update_meta(m) if m and self._iface else None

    def _op(self, d, f, k, v) -> None:
        if isinstance(v, File):
            if isinstance(v, Image):
                v.load(self.settings.get_dir())
            # TODO: add step to serialise data for files
            v._mkcopy(self.settings.get_dir())  # key independent
            # d[k] = int(v._id, 16)
            if k not in f:
                f[k] = [v]
            else:
                f[k].append(v)
        elif isinstance(v, (int, float)):
            d[k] = v
        else:
            pass  # raise not supported error
        return d, f

    def _finish(self, exit_code) -> None:
        self._monitor.stop()
