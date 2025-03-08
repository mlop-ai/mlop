import json
import logging
import queue
import threading
import time

import httpx

from .sets import Settings

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Interface"


class ServerInterface:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.url_view = f"{self.settings.url_view}/{self.settings.user}/{self.settings.project}/{self.settings._op_id}"
        self.headers = {
            "Authorization": f"Bearer {self.settings.auth}",
            "Content-Type": "application/json",
            "User-Agent": f"{__name__.split('.')[0]}",
            "X-Run-Id": f"{self.settings._op_id}",
            "X-Run-Name": f"{self.settings._op_name}",
            "X-Project-Name": f"{self.settings.project}",
        }
        self.headers_data = self.headers.copy()
        self.headers_data.update({"Content-Type": "application/x-ndjson"})

        self.client = httpx.Client(
            # http2=True,
            verify=False,  # TODO: enable ssl
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
            verify=False,  # TODO: enable ssl
            proxy=self.settings.http_proxy or self.settings.https_proxy or None,
        )

        self._stop_event = threading.Event()

        self._queue_data = queue.Queue()
        self._buffer_data = None
        self._thread_data = None
        self._thread_file = None
        self._thread_storage = None

        self._queue_message = self.settings.message
        self._buffer_message = None
        self._thread_message = None

    def start(self) -> None:
        if self._thread_data is None:
            self._thread_data = threading.Thread(
                target=self._worker_publish,
                args=(
                    self.settings.url_data,
                    self.headers_data,
                    self._queue_data,
                    self._buffer_data,
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
                    self._buffer_message,
                    self._stop_event.is_set,
                    "message" if self.settings.mode == "debug" else None,
                ),
                daemon=True,
            )
            self._thread_message.start()
        r = self._post_v1(
            self.settings.url_status,
            self.headers,
            make_compat_status_v1(self.settings.system.info(), self.settings),
            client=self.client,
        )
        try:
            logger.info(f"{tag}: started: {r.json()}") # TODO: send a proper response
        except Exception as e:
            logger.critical("%s: failed to start: %s", tag, e)

    def publish(
        self,
        data: dict[str, any] | None = None,
        file: None = None,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
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
        for t in [
            self._thread_data,
            self._thread_file,
            self._thread_storage,
            self._thread_message,
        ]:
            if t is not None:
                t.join(timeout=self.settings.x_internal_check_process)
                t = None
        logger.info(f"{tag}: find uploaded data at {self.url_view}")

    def _worker_publish(self, e, h, q, b, stop, name=None):
        while not q.empty() or not stop():
            if not q.empty():
                _ = self._post_v1(
                    e,
                    h,
                    q,
                    b,
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
            logger.info(f"{tag}: file api responded {len(d)} key(s)")
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
            logger.critical("%s: failed to send files: %s", tag, e)

    def _queue_iter(self, q, b):
        s = time.time()
        while (
            len(b) < self.settings.x_file_stream_max_size
            and (time.time() - s) < self.settings.x_file_stream_transmit_interval
        ):
            try:
                v = q.get(block=False)
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
                # logger.info(f"{tag}: put a file in storage")
                return r
            else:
                logger.error(
                    f"{tag}: server responded error {r.status_code} during PUT: {r.text}"
                )
        except Exception as e:
            logger.error("%s: no response received: %s", tag, e)
        retry += 1
        self._put_v1(
            url, headers, content, client=client, retry=retry
        ) if retry < self.settings.x_file_stream_retry_max else logger.critical(
            f"{tag}: failed to put file in storage after {retry} retries"
        )

    def _post_v1(self, url, headers, q, b=[], client=None, name=None, retry=0):
        b = []
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
                    logger.info(
                        f"{tag}: {name}: sent {len(b)} line(s) at {len(b) / (time.time() - s):.2f} lines/s"
                    )
                return r
            else:
                if name is not None:
                    logger.error(
                        f"{tag}: {name}: server responded error {r.status_code} for {len(b)} line(s) during POST: {r.text}"
                    )
        except Exception as e:
            logger.error("%s: no response received during POST: %s", tag, e)

        retry += 1
        if retry < self.settings.x_file_stream_retry_max:
            logger.warning(
                f"{tag}: retry {retry}/{self.settings.x_file_stream_retry_max} for {len(b)} line(s)"
            )
            time.sleep(
                min(
                    self.settings.x_file_stream_retry_wait_min_seconds * (2**retry),
                    self.settings.x_file_stream_retry_wait_max_seconds,
                )
            )
            for i in b:  # if isinstance(q, queue.Queue)
                q.put(i, block=False)
            return self._post_v1(
                url, headers, q, b, client=client, retry=retry
            )  # kwargs
        else:
            if name is not None:
                logger.critical(
                    f"{tag}: {name}: failed to send {len(b)} line(s) after {retry} retries"
                )
            return None


def make_compat_data_v1(data, timestamp, step):
    line = [
        json.dumps(
            {
                "time": int(timestamp * 1000),  # convert to ms
                "step": int(step),
                "data": data,
            }
        )
    ]
    return ("\n".join(line) + "\n").encode("utf-8")


def make_compat_file_v1(file, timestamp, step):
    batch = []
    for k, fl in file.items():
        for f in fl:
            i = {
                "fileName": f"{f._name}{f._ext}",
                "size": f._size,
                "fileType": f._ext[1:],
                "logName": k,
                "step": step,
            }
            batch.append(i)
    return json.dumps({"files": batch}).encode()


def make_compat_storage_v1(f, fl):
    # workaround for lack of file ident on server side
    for i in fl:
        if next(iter(i.keys())) == f"{f._name}{f._ext}":
            return next(iter(i.values()))
    return None


def make_compat_message_v1(level, message, timestamp, step):
    line = [
        json.dumps(
            {
                "time": int(timestamp * 1000),  # convert to ms
                "message": message,
                "lineNumber": step,
                "logType": "INFO" if level == logging.INFO else "ERROR",
            }
        )
    ]
    return ("\n".join(line) + "\n").encode("utf-8")


def make_compat_status_v1(data, settings):
    return json.dumps(
        {
            "runId": settings._op_id,
            "runName": settings._op_name,
            "projectName": settings.project,
            "metadata": json.dumps(data),
        }
    ).encode()
