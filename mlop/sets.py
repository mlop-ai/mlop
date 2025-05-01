import logging
import os
import queue
import sys
from typing import Any, Dict, List

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Setup"


class Settings:
    tag: str = f"{__name__.split('.')[0]}"
    dir: str = str(os.path.abspath(os.getcwd()))

    _auth: str = None
    _sys: Dict[str, Any] = {}
    compat: Dict[str, Any] = {}
    project: str = "default"
    mode: str = "perf"  # noop | debug | perf
    meta: List[Any] = None
    message: queue.Queue = queue.Queue()
    disable_store: bool = True  # TODO: make false
    disable_iface: bool = False
    disable_console: bool = False  # disable file-based logging

    _op_name: str = None
    _op_id: int = None
    _op_status: int = -1

    store_db: str = "store.db"
    store_table_num: str = "num"
    store_table_file: str = "file"
    store_max_size: int = 2**14
    store_aggregate_interval: float = 2 ** (-1)

    http_proxy: str = None
    https_proxy: str = None
    insecure_disable_ssl: bool = False

    x_log_level: int = 2**4  # logging.NOTSET
    x_internal_check_process: int = 1  # TODO: make configurable
    x_file_stream_retry_max: int = 2**2
    x_file_stream_retry_wait_min_seconds: float = 2 ** (-1)
    x_file_stream_retry_wait_max_seconds: float = 2
    x_file_stream_timeout_seconds: int = 2**5  # 2**2
    x_file_stream_max_conn: int = 2**5
    x_file_stream_max_size: int = 2**18
    x_file_stream_transmit_interval: int = 2**3
    x_sys_sampling_interval: int = 2**2
    x_sys_label: str = "sys"
    x_grad_label: str = "grad"
    x_param_label: str = "param"

    host: str = None
    url_view: str = None
    url_webhook: str = None

    def update(self, settings) -> None:
        if isinstance(settings, Settings):
            settings = settings.to_dict()
        for key, value in settings.items():
            setattr(self, key, value)
        self.update_host()

    def update_host(self):
        if self.host is not None:
            self.url_app = f"http://{self.host}:3000"
            self.url_api = f"http://{self.host}:3001"
            self.url_ingest = f"http://{self.host}:3003"
            self.url_py = f"http://{self.host}:3004"
        elif not (  # backwards compatibility
            hasattr(self, "url_app")
            and hasattr(self, "url_api")
            and hasattr(self, "url_ingest")
            and hasattr(self, "url_py")
        ):
            self.url_app = "https://app.mlop.ai"
            self.url_api = "https://api-prod.mlop.ai"
            self.url_ingest = "https://ingest-prod.mlop.ai"
            self.url_py = "https://py-prod.mlop.ai"
        self.update_url()

    def update_url(self):
        self.url_token = f"{self.url_app}/api-keys"
        self.url_login = f"{self.url_api}/api/slug"
        self.url_start = f"{self.url_api}/api/runs/create"
        self.url_stop = f"{self.url_api}/api/runs/status/update"
        self.url_meta = f"{self.url_api}/api/runs/logName/add"
        self.url_graph = f"{self.url_api}/api/runs/modelGraph/create"
        self.url_num = f"{self.url_ingest}/ingest/metrics"
        self.url_data = f"{self.url_ingest}/ingest/data"
        self.url_file = f"{self.url_ingest}/files"
        self.url_message = f"{self.url_ingest}/ingest/logs"
        self.url_alert = f"{self.url_py}/api/runs/alert"
        self.url_trigger = f"{self.url_py}/api/runs/trigger"

    def to_dict(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.__annotations__.keys()}

    def get_dir(self) -> str:
        return os.path.join(
            self.dir,
            "." + self.tag,
            self.project,
            self._op_name,  # str(self._op_id)
        )

    def _nb(self) -> bool:
        return (
            get_console() in ["ipython", "jupyter"]
            or self._nb_colab()
            or self._nb_kaggle()
        )

    def _nb_colab(self) -> bool:
        return "google.colab" in sys.modules

    def _nb_kaggle(self) -> bool:
        return (
            os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None
            or "kaggle_environments" in sys.modules
            or "kaggle" in sys.modules
        )


def get_console() -> str:
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return "python"
    except ImportError:
        return "python"

    if "spyder" in sys.modules or "terminal" in ipython.__module__:
        return "ipython"

    connection_file = (
        ipython.config.get("IPKernelApp", {}).get("connection_file", "")
        or ipython.config.get("ColabKernelApp", {}).get("connection_file", "")
    ).lower()
    if "jupyter" not in connection_file:
        return "ipython"
    else:
        return "jupyter"
