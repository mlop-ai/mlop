import logging
import queue
import threading
import time

import httpx
import keyring
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .api import (
    make_compat_data_v1,
    make_compat_file_v1,
    make_compat_meta_v1,
    make_compat_num_v1,
    make_compat_start_v1,
    make_compat_stop_v1,
    make_compat_storage_v1,
)
from .sets import Settings
from .util import print_url

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Interface"


class ServerInterface:
    def __init__(self, config: dict, settings: Settings) -> None:
        self.config = config
        self.settings = settings
        self.settings._auth = keyring.get_password(
            f"{self.settings.tag}", f"{self.settings.tag}"
        )

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
            verify=True if not self.settings.insecure_disable_ssl else False,
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
            verify=True if not self.settings.insecure_disable_ssl else False,
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
        )
        self._progress_task = None
        self._thread_progress = None
        self._lock_progress = threading.Lock()
        self._nb = bool(self.settings._nb())
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
        if self._thread_progress is None:
            self._thread_progress = threading.Thread(
                target=self._worker_progress, daemon=True
            )
            self._thread_progress.start()

    def publish(
        self,
        num: dict[str, any] | None = None,
        data: dict[str, any] | None = None,
        file: dict[str, any] | None = None,
        timestamp: int | None = None,
        step: int | None = None,
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

    def stop(self) -> None:
        self._stop_event.set()

        while (
            not self._queue_num.empty()
            or not self._queue_data.empty()
            or not self._queue_message.empty()
        ):
            time.sleep(self.settings.x_internal_check_process / 10)  # TODO: cleanup

        if self._progress_task is not None:
            self._progress.stop()
            self._progress_task = None

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
        self._update_status(self.settings)
        logger.info(f"{tag}: find uploaded data at {print_url(self.settings.url_view)}")

    def _update_status(self, settings, trace=None):
        r = self._post_v1(
            self.settings.url_stop,
            self.headers,
            make_compat_stop_v1(self.settings, trace),
            client=self.client_api,
        )

    def _update_meta(self, num=None, df=None):
        self._thread_meta = threading.Thread(
            target=self._worker_meta, args=(num, df), daemon=True
        )
        self._thread_meta.start()

    def _worker_progress(self):
        while not (
            self._stop_event.is_set()
            and self._queue_num.empty()
            and self._queue_data.empty()
            and self._queue_message.empty()
        ):
            with self._lock_progress:
                if self._total > 0:
                    if self._progress_task is None:
                        self._progress_task = self._progress.add_task(
                            "Processing", total=100
                        )
                        self._progress.start()

                    if self._total > 0:
                        i = self._total - (
                            self._queue_num.qsize()
                            + self._queue_data.qsize()
                            + self._queue_message.qsize()
                        )
                        self._progress.update(
                            self._progress_task,
                            completed=min(100 * i / self._total, 100),
                            description=f"Uploading ({max(i, 0)}/{self._total}):",
                        )
                        if self._nb and hasattr(self._progress, 'live'):
                            self._progress.live.refresh()

            time.sleep(self.settings.x_internal_check_process)

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

    def _worker_storage(self, f):
        _ = self._put_v1(
            f._url,
            {
                "Content-Type": f._type,  # "application/octet-stream"
            },
            open(f._path, "rb"),
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
            logger.debug(f"{tag}: file api responded {len(d)} key(s)")
            for k, fel in file.items():
                for f in fel:
                    f._url = make_compat_storage_v1(f, d[k])
                    if not f._url:
                        logger.critical(f"{tag}: file api did not provide storage url")
                    else:
                        self._thread_storage = threading.Thread(
                            target=self._worker_storage, args=(f,), daemon=True
                        )
                        self._thread_storage.start()
        except Exception as e:
            logger.critical(
                "%s: failed to send files to %s: %s (%s)",
                tag,
                self.settings.url_file,
                e,
                type(e).__name__,
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

    def _put_v1(self, url, headers, content, client=None, retry=0):
        try:
            r = client.put(
                url,
                content=content,
                headers=headers,
            )
            if r.status_code in [200, 201]:
                logger.debug(f"{tag}: successfully put one item in storage")
                return r
            else:
                logger.error(
                    f"{tag}: server responded error {r.status_code} during PUT to {url}: {r.text}"
                )
        except Exception as e:
            logger.error(
                "%s: no response received during PUT to %s: %s: %s",
                tag,
                url,
                type(e).__name__,
                e,
            )
        retry += 1
        self._put_v1(
            url, headers, content, client=client, retry=retry
        ) if retry < self.settings.x_file_stream_retry_max else logger.critical(
            f"{tag}: failed to put item in storage after {retry} retries to {url}"
        )

    def _post_v1(self, url, headers, q, client=None, name=None, retry=0):
        b, r = [], None
        try:
            s = time.time()
            content = self._queue_iter(q, b) if isinstance(q, queue.Queue) else q
            r = client.post(
                url,
                content=content,  # iter(q.get, None),
                headers=headers,
            )
            if r.status_code in [200, 201]:
                if name is not None and isinstance(q, queue.Queue):
                    logger.debug(
                        f"{tag}: {name}: sent {len(b)} line(s) at {len(b) / (time.time() - s):.2f} lines/s to {url}"
                    )
                return r
            else:
                if name is not None:
                    logger.error(
                        f"{tag}: {name}: server responded error {r.status_code} for {len(b)} line(s) during POST: {r.text}"
                    )
        except Exception as e:
            logger.error(
                "%s: no response received during POST to %s: %s: %s",
                tag,
                url,
                type(e).__name__,
                e,
            )

        retry += 1
        if retry < self.settings.x_file_stream_retry_max:
            logger.warning(
                f"{tag}: retry {retry}/{self.settings.x_file_stream_retry_max}: server responded error {r.status_code} for {len(b)} line(s) during POST to {url}: {r.text}"
                if r
                else f"{tag}: retry {retry}/{self.settings.x_file_stream_retry_max}: no response received for {len(b)} line(s) during POST to {url}"
            )
            time.sleep(
                min(
                    self.settings.x_file_stream_retry_wait_min_seconds * (2**retry),
                    self.settings.x_file_stream_retry_wait_max_seconds,
                )
            )
            for i in b:  # if isinstance(q, queue.Queue)
                q.put(i, block=False)
            return self._post_v1(url, headers, q, client=client, retry=retry)  # kwargs
        else:
            if name is not None:
                logger.critical(
                    f"{tag}: {name}: failed to send {len(b)} line(s) after {retry} retries"
                )
            return None
