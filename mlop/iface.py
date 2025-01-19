import json
import logging
import queue
import threading
import time

import httpx

from .file import dict_to_json
from .sets import Settings

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Interface"


class ServerInterface:
    def __init__(self, settings: Settings) -> None:
        self.url_data = settings.url_data
        self.url_view_op = (
            f"{settings.url_view}/{settings.user}/{settings.project}/{settings._op_id}"
        )

        self.headers = {
            "Authorization": f"Bearer {settings.auth}",
            "User-Agent": f"{__name__.split('.')[0]}",
            "X-Run-Id": f"{settings._op_id}",
            "X-Run-Name": f"{settings._op_name}",
            "X-Project-Name": f"{settings.project}",
        }

        self.client = httpx.Client(
            http2=True,
            verify=False,
            proxy=settings.http_proxy or settings.https_proxy or None,
            limits=httpx.Limits(
                max_keepalive_connections=settings.x_file_stream_max_conn,
                max_connections=settings.x_file_stream_max_conn,
            ),
            timeout=httpx.Timeout(
                settings.x_file_stream_timeout_seconds,
                # connect=settings.x_file_stream_timeout_seconds,
            ),
        )

        self.max_size = settings.x_file_stream_max_size
        self.retry_max = settings.x_file_stream_retry_max
        self.retry_wait_min = settings.x_file_stream_retry_wait_min_seconds
        self.retry_wait_max = settings.x_file_stream_retry_wait_max_seconds
        self.transmit_interval = settings.x_file_stream_transmit_interval

        self._wait = settings.x_internal_check_process
        self._stop_event = threading.Event()
        self._queue_data = queue.Queue()
        self._thread_publish = None

    def start(self) -> None:
        if self._thread_publish is None:
            self._thread_publish = threading.Thread(
                target=self._worker_publish, daemon=True
            )
            self._thread_publish.start()

    def publish(
        self,
        data: dict[str, any] | None = None,
        file: None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        if data:
            data = dict_to_json(data)  # make safer
            self._queue_data.put(
                make_compat_data_v1(data, timestamp, step), block=False
            )
        elif file:
            pass  # self._queue_file.put((file, timestamp, step))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread_publish is not None:
            self._thread_publish.join(timeout=self._wait)
            self._thread_publish = None
        logger.info(f"{tag}: find uploaded data at {self.url_view_op}")

    def _worker_publish(self) -> None:
        while not self._stop_event.is_set() or not self._queue_data.empty():
            if not self._queue_data.empty():
                headers = self.headers
                headers["Content-Type"] = "application/x-ndjson"
                _ = self._post_v1(
                    self.url_data,
                    headers,
                    self._queue_data,
                    client=self.client,
                )

    def _queue_iter(self, q):
        s = time.time()
        while (
            len(self._queue_buffer) < self.max_size
            and (time.time() - s) < self.transmit_interval
        ):
            try:
                v = q.get(block=False)
                self._queue_buffer.append(v)
                yield v
            except queue.Empty:
                break

    def _post_v1(self, url, headers, q, client=None, retry=0):
        self._queue_buffer = []
        try:
            s = time.time()
            r = client.post(
                url,
                content=self._queue_iter(q),  # self._queue_iter(q) # iter(q.get, None),
                headers=headers,
            )
            if r.status_code in [200, 201]:
                logger.info(
                    f"{tag}: sent {len(self._queue_buffer)} item(s) at {len(self._queue_buffer) / (time.time() - s):.2f} items/s"
                )
                return r
            else:
                logger.error(
                    f"{tag}: server responded error {r.status_code} for {len(self._queue_buffer)} item(s): {r.text}"
                )
        except Exception as e:
            logger.error("%s: no response received: %s", tag, e)

        retry += 1
        if retry < self.retry_max:
            logger.warning(
                f"{tag}: retry {retry}/{self.retry_max} for {len(self._queue_buffer)} item(s)"
            )
            time.sleep(min(self.retry_wait_min * (2**retry), self.retry_wait_max))
            for i in self._queue_buffer:
                q.put(i, block=False)
            return self._post_v1(url, headers, q, client=client, retry=retry)
        else:
            logger.critical(f"{tag}: failed to send {len(self._queue_buffer)} item(s)")
            return None


def make_compat_data_v1(data, timestamp, step):
    batch = []
    for k, v in data.items():  # future compatibility
        i = {
            "time": int(timestamp * 1000),  # convert to ms
            "step": int(step),
            "data": {
                k: v,
            },
        }
        batch.append(json.dumps(i))
    return ("\n".join(batch) + "\n").encode("utf-8")
