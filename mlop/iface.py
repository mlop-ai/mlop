import logging
import queue
import threading
import time
from typing import Any, Dict, List, Union

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .api import (
    make_compat_data_v1,
    make_compat_file_v1,
    make_compat_meta_v1,
    make_compat_num_v1,
    make_compat_start_v1,
    make_compat_status_v1,
    make_compat_storage_v1,
)
from .log import _stderr
from .sets import Settings
from .util import print_url

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Interface"


class ServerInterface:
    def __init__(self, config: dict, settings: Settings) -> None:
        self.config = config
        self.settings = settings

        # self.url_view = f"{self.settings.url_view}/{self.settings.user}/{self.settings.project}/{self.settings._op_id}"
        self.headers = {
            "Authorization": f"Bearer {self.settings._auth}",
            "Content-Type": "application/json",
            "User-Agent": f"{self.settings.tag}",
            "X-Run-Id": f"{self.settings._op_id}",
            "X-Run-Name": f"{self.settings._op_name}",
            "X-Project-Name": f"{self.settings.project}",
        }
        self.headers_num = self.headers.copy()
        self.headers_num.update({"Content-Type": "application/x-ndjson"})

        self.client = httpx.Client(
            verify=not self.settings.insecure_disable_ssl,
            proxy=self.settings.http_proxy or self.settings.https_proxy or None,
            limits=httpx.Limits(
                max_keepalive_connections=self.settings.x_file_stream_max_conn,
                max_connections=self.settings.x_file_stream_max_conn,
            ),
            timeout=httpx.Timeout(
                self.settings.x_file_stream_timeout_seconds,
                # connect=settings.x_file_stream_timeout_seconds,
            ),
        )
        self.client_storage = httpx.Client(
            # http1=False, # TODO: set http2
            verify=not self.settings.insecure_disable_ssl,
            proxy=self.settings.http_proxy or self.settings.https_proxy or None,
            timeout=httpx.Timeout(self.settings.x_file_stream_timeout_seconds),
        )
        self.client_api = httpx.Client(
            verify=not self.settings.insecure_disable_ssl,
            proxy=self.settings.http_proxy or self.settings.https_proxy or None,
            timeout=httpx.Timeout(
                self.settings.x_file_stream_timeout_seconds,
            ),
        )

        self._stop_event = threading.Event()

        self._queue_num = queue.Queue()
        self._thread_num = None
        self._queue_data = queue.Queue()
        self._thread_data = None
        self._thread_file = None
        self._thread_storage = None
        self._thread_meta = None

        self._queue_message = self.settings.message
        self._thread_message = None

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
            console=Console(file=_stderr),
            # redirect_stdout=False,
        )
        self._progress_task = None
        self._thread_progress = None
        self._lock_progress = threading.Lock()
        self._total = 0

    def start(self) -> None:
        logger.info(f"{tag}: find live updates at {print_url(self.settings.url_view)}")
        if self._thread_num is None:
            self._thread_num = threading.Thread(
                target=self._worker_publish,
                args=(
                    self.settings.url_num,
                    self.headers_num,
                    self._queue_num,
                    self._stop_event.is_set,
                    "num",
                ),
                daemon=True,
            )
            self._thread_num.start()
        if self._thread_data is None:
            self._thread_data = threading.Thread(
                target=self._worker_publish,
                args=(
                    self.settings.url_data,
                    self.headers,
                    self._queue_data,
                    self._stop_event.is_set,
                    "data",
                ),
                daemon=True,
            )
            self._thread_data.start()
        if self._thread_message is None:
            self._thread_message = threading.Thread(
                target=self._worker_publish,
                args=(
                    self.settings.url_message,
                    self.headers,
                    self._queue_message,
                    self._stop_event.is_set,
                    "message" if self.settings.mode == "debug" else None,
                ),
                daemon=True,
            )
            self._thread_message.start()
        if self._thread_progress is None and not self.settings.disable_progress:
            self._thread_progress = threading.Thread(
                target=self._worker_progress, daemon=True
            )
            self._thread_progress.start()

    def publish(
        self,
        num: Union[Dict[str, Any], None] = None,
        data: Union[Dict[str, Any], None] = None,
        file: Union[Dict[str, Any], None] = None,
        timestamp: Union[int, None] = None,
        step: Union[int, None] = None,
    ) -> None:
        with self._lock_progress:  # enforce one thread at a time
            self._total += 1
            if num:
                self._queue_num.put(
                    make_compat_num_v1(num, timestamp, step), block=False
                )
            if data:
                self._queue_data.put(
                    make_compat_data_v1(data, timestamp, step), block=False
                )
            if file:
                self._thread_file = threading.Thread(
                    target=self._worker_file,
                    args=(file, make_compat_file_v1(file, timestamp, step)),
                    daemon=True,
                )  # TODO: batching
                self._thread_file.start()

    def save(self) -> None:
        while not self._queue_num.empty() or not self._queue_data.empty():
            time.sleep(self.settings.x_internal_check_process / 10)  # TODO: cleanup

    def stop(self) -> None:
        if self._thread_progress is None:
            self._thread_progress = threading.Thread(
                target=self._worker_progress, daemon=True
            )
            self._thread_progress.start()

        self._stop_event.set()
        self.save()

        for t in [
            self._thread_num,
            self._thread_data,
            self._thread_file,
            self._thread_storage,
            self._thread_message,
            self._thread_meta,
            self._thread_progress,
        ]:
            if t is not None:
                t.join(timeout=None)
                t = None

        if self._progress_task is not None:
            self._progress.remove_task(self._progress_task)
            self._progress_task = None
        self._progress.stop()
        self._update_status(self.settings)

        logger.info(
            f"{tag}: find {self._total} synced entries at {print_url(self.settings.url_view)}"
        )
        if self.settings.meta and self.settings.mode == "debug":
            logger.info(f"{tag}: recorded metadata:")
            for e in sorted(self.settings.meta, key=len):
                logger.info(f"    {e}")

    def _update_status(self, settings, trace: Union[Any, None] = None):
        r = self._post_v1(
            self.settings.url_stop,
            self.headers,
            make_compat_status_v1(self.settings, trace),
            client=self.client_api,
        )

    def _update_meta(
        self,
        num: Union[List[str], None] = None,
        df: Union[Dict[str, List[str]], None] = None,
    ):
        self._thread_meta = threading.Thread(
            target=self._worker_meta, args=(num, df), daemon=True
        )
        self._thread_meta.start()

    def _worker_progress(self):
        while not (
            self._stop_event.is_set()
            and self._queue_num.empty()
            and self._queue_data.empty()
        ):
            with self._lock_progress:
                if self._total > 0:
                    i = self._total - (
                        self._queue_num.qsize() + self._queue_data.qsize()
                    )
                    p = 100 * i / self._total

                    if self._progress_task is None and p < 100:  # init
                        self._progress_task = self._progress.add_task(
                            "Processing:", total=100
                        )
                        self._progress.start()

                    if self._progress_task is not None:  # update if exists
                        self._progress.update(
                            self._progress_task,
                            completed=min(p, 100),
                            description=f"Uploading ({max(i, 0)}/{self._total}):",
                        )
                        if p >= 100:
                            self._progress.remove_task(self._progress_task)
                            self._progress_task = None  # signal no active task

            time.sleep(self.settings.x_internal_check_process / 2)

    def _worker_publish(self, e, h, q, stop, name=None):
        while not (q.empty() and stop()):  # terminates only when both conditions met
            if q.empty():
                time.sleep(self.settings.x_internal_check_process)  # debounce
            else:
                _ = self._post_v1(
                    e,
                    h,
                    q,
                    client=self.client,
                    name=name,
                )

    def _worker_storage(self, f, url, data):
        _ = self._put_v1(
            url,
            {
                "Content-Type": f._type,  # "application/octet-stream"
            },
            data,
            client=self.client_storage,
        )

    def _worker_file(self, file, q):
        r = self._post_v1(
            self.settings.url_file,
            self.headers,
            q,
            client=self.client,
        )
        try:
            d = r.json()
            for k, fel in file.items():
                for f in fel:
                    url = make_compat_storage_v1(f, d[k])
                    with open(
                        f._path, "rb"
                    ) as file:  # TODO: data = open(f._path, "rb")
                        data = file.read()
                    if not url:
                        logger.critical(f"{tag}: file api did not provide storage url")
                    else:
                        self._thread_storage = threading.Thread(
                            target=self._worker_storage,
                            args=(f, url, data),
                            daemon=True,
                        )
                        self._thread_storage.start()
        except Exception as e:
            logger.critical(
                "%s: failed to send files to %s: [%s] %s",
                tag,
                self.settings.url_file,
                type(e).__name__,
                e,
            )

    def _worker_meta(self, num=None, file=None):
        if num:
            r = self._post_v1(
                self.settings.url_meta,
                self.headers,
                make_compat_meta_v1(num, "num", self.settings),
                client=self.client_api,
            )
        if file:
            for k, v in file.items():
                r = self._post_v1(
                    self.settings.url_meta,
                    self.headers,
                    make_compat_meta_v1(v, k, self.settings),
                    client=self.client_api,
                )

    def _queue_iter(self, q, b):
        s = time.time()
        while (
            len(b) < self.settings.x_file_stream_max_size
            and (time.time() - s) < self.settings.x_file_stream_transmit_interval
        ):
            try:
                v = q.get(timeout=self.settings.x_internal_check_process)
                b.append(v)
                yield v
            except queue.Empty:
                break

    def _try(
        self,
        method,
        url,
        headers,
        content,
        name: Union[str, None] = None,
        q: Union[queue.Queue, None] = None,
        retry=0,
    ):
        if retry >= self.settings.x_file_stream_retry_max:
            logger.critical(f"{tag}: {name}: failed after {retry} retries")
            return None

        try:
            r = method(url, content=content, headers=headers)
            if r.status_code in [200, 201]:
                return r
            logger.warning(
                f"{tag}: {name}: retry {retry + 1}/{self.settings.x_file_stream_retry_max}: response code {r.status_code if r else 'N/A'} for {len(q) if q else 'request'} from {url}: {r.text if r else 'N/A'}"
            )
        except Exception as e:
            logger.debug(
                "%s: %s: retry %s/%s: no response from %s: %s: %s",
                tag,
                name,
                retry + 1,
                self.settings.x_file_stream_retry_max,
                url,
                type(e).__name__,
                e,
            )
        time.sleep(
            min(
                self.settings.x_file_stream_retry_wait_min_seconds * (2 ** (retry + 1)),
                self.settings.x_file_stream_retry_wait_max_seconds,
            )
        )

        if q is not None:  # requeue items
            for i in q:
                content.put(i, block=False)
        return self._try(method, url, headers, content, name=name, q=q, retry=retry + 1)

    def _put_v1(self, url, headers, content, client, name="put"):
        return self._try(
            client.put,
            url,
            headers,
            content,
            name=name,
        )

    def _post_v1(self, url, headers, q, client, name: Union[str, None] = "post"):
        b, r = [], None
        content = self._queue_iter(q, b) if isinstance(q, queue.Queue) else q

        s = time.time()
        r = self._try(
            client.post,
            url,
            headers,
            content,
            name=name,
            q=b if isinstance(q, queue.Queue) else None,
        )

        if (
            r
            and r.status_code in [200, 201]
            and name is not None
            and isinstance(q, queue.Queue)
        ):
            logger.debug(
                f"{tag}: {name}: sent {len(b)} line(s) at {len(b) / (time.time() - s):.2f} lines/s to {url}"
            )
        return r
