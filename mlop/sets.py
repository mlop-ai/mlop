import logging
import os
import queue
import sys

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Setup"


class Settings:
    tag: str = f"{__name__.split('.')[0]}"
    dir: str = str(os.path.abspath(os.getcwd()))

    auth: str = None
    project: str = "default"
    mode: str = "perf"  # noop | debug | perf
    system: dict[str, any] = {}
    meta: list = None
    message: queue.Queue = queue.Queue()
    disable_store: bool = True  # TODO: make false
    disable_iface: bool = False
    disable_logger: bool = False  # disable file-based logging

    _op_name: str = None
    _op_id: int = None
    _op_status: int = 1

    store_db: str = "store.db"
    store_table_data: str = "data"
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
    x_sys_label: str = "_sys/"
    x_meta_label: str = "_meta/"

    url: str = "https://demo.mlop.ai"
    url_token: str = f"{url}/api-keys"
    _url_api: str = "https://api.mlop.ai"
    url_login: str = f"{_url_api}/api/is-logged-in"
    url_start: str = f"{_url_api}/api/create-run"
    url_stop: str = f"{_url_api}/api/update-run-status"
    url_meta: str = f"{_url_api}/api/add-logname"
    _url: str = "https://server.mlop.ai"
    url_data: str = f"{_url}/ingest/metrics"
    url_file: str = f"{_url}/files"
    url_message: str = f"{_url}/ingest/logs"
    url_view: str = f"{_url}/view"

    def update(self, settings) -> None:
        if isinstance(settings, Settings):
            settings = settings.to_dict()
        for key, value in settings.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, any]:
        return {key: getattr(self, key) for key in self.__annotations__.keys()}

    def get_dir(self) -> str:
        return os.path.join(
            self.dir, "." + self.tag, self.project, self._op_name, str(self._op_id)
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


class OpsSetup:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings


def setup(settings: Settings | None = None) -> OpsSetup:
    logger.debug(f"{tag}: loading settings")
    return OpsSetup(settings=settings)


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
