import importlib.metadata
import logging
import os
import platform
import time

import psutil
from git import Repo

from .sets import Settings
from .util import to_human  # TODO: move to server side

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "System"


class System:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.uname = platform.uname()._asdict()
        self.timezone = list(time.tzname)

        self.cpu_count = psutil.cpu_count
        self.cpu_freq = [i._asdict() for i in psutil.cpu_freq(percpu=True)]
        self.cpu_freq_min = min([i["min"] for i in self.cpu_freq])
        self.cpu_freq_max = max([i["max"] for i in self.cpu_freq])
        self.svmem = psutil.virtual_memory()._asdict()
        self.sswap = psutil.swap_memory()._asdict()
        self.disk = [i._asdict() for i in psutil.disk_partitions()]
        self.net_if_addrs = {
            i: [
                {k: v for k, v in j._asdict().items() if k != "family"}
                for j in psutil.net_if_addrs()[i]
            ]
            for i in psutil.net_if_addrs()
        }
        self.boot_time = psutil.boot_time()
        self.users = [i._asdict() for i in psutil.users()]

        self.pid = os.getpid()
        self.proc = psutil.Process(pid=self.pid)
        with self.proc.oneshot():  # perf
            self.proc_info = self.proc.as_dict(attrs=["exe", "cmdline"])
            self.proc_child = self.proc.children(recursive=True)
            self.pid_child = [p.pid for p in self.proc_child] + [self.pid]
        if self.settings.mode == "debug":  # privacy guard
            self.environ = self.proc.environ()
            self.requirements = [
                f"{p.metadata['Name']}=={p.version}"
                for p in importlib.metadata.distributions()
            ]

        self.gpu = self.get_gpu()
        self.git = self.get_git()

    def __getattr__(self, name):
        return self.get_psutil(name)

    def get_psutil(self, name):  # handling os specific methods
        if hasattr(psutil, name):
            return getattr(psutil, name)
        else:
            return None

    def get_gpu(self):
        d = {}
        try:  # NVIDIA
            import pynvml

            try:
                pynvml.nvmlInit()
                logger.info(f"{tag}: NVIDIA GPU detected")
                d["nvidia"] = {
                    "count": pynvml.nvmlDeviceGetCount(),
                    "driver": pynvml.nvmlSystemGetDriverVersion(),
                    "devices": [],
                    "handles": [],
                }
                for i in range(d["nvidia"]["count"]):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    d["nvidia"]["handles"].append(h)
                    d["nvidia"]["devices"].append(
                        {
                            "name": pynvml.nvmlDeviceGetName(h),
                            "memory": {
                                "total": to_human(
                                    pynvml.nvmlDeviceGetMemoryInfo(h).total
                                ),
                            },
                            "temp": pynvml.nvmlDeviceGetTemperature(
                                h, pynvml.NVML_TEMPERATURE_GPU
                            ),
                            "pid": [
                                p.pid
                                for p in (
                                    pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                                    + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
                                )
                            ],
                        }
                    )
            except pynvml.NVMLError_LibraryNotFound:
                logger.debug(f"{tag}: NVIDIA: driver not found")
            except Exception as e:
                logger.error("%s: NVIDIA: error: %s", tag, e)
        except ImportError:
            logger.debug(f"{tag}: NVIDIA: pynvml not found")
        return d

    def get_git(self):
        d = {}
        try:
            repo = Repo(
                f"{self.settings.work_dir()}" or os.getcwd(),
                search_parent_directories=True,
            )
            try:
                url = repo.remotes["origin"].url
            except IndexError:
                url = None
            d = {
                "url": url,
                # "name": repo.config_reader().get_value("user", "name"),
                "email": repo.config_reader().get_value("user", "email"),
                "root": repo.git.rev_parse("--show-toplevel"),
                "dirty": repo.is_dirty(),
                "branch": repo.head.ref.name,
                "commit": repo.head.commit.hexsha,
            }
        except Exception as e:
            logger.debug("%s: git: repository not detected: %s", tag, e)
        return d

    def info(self):
        d = {
            "process": {
                **self.proc_info,
                "pid": self.pid,
            },
            "platform": self.uname,
            "timezone": self.timezone,
            "cpu": {
                "physical": self.cpu_count(logical=False),
                "virtual": self.cpu_count(logical=True),
                "freq": {
                    "min": self.cpu_freq_min,
                    "max": self.cpu_freq_max,
                },
            },
            "memory": {
                "virt": to_human(self.svmem["total"]),
                "swap": to_human(self.sswap["total"]),
            },
            "boot_time": self.boot_time,
        }
        if self.gpu:
            d["gpu"] = {}
            if self.gpu.get("nvidia"):
                d["gpu"]["nvidia"] = {
                    k: v for k, v in self.gpu["nvidia"].items() if k != "handles"
                }
        if self.git:
            d["git"] = self.git
        if self.settings.mode == "debug":
            d["process"]["environ"] = self.environ
            d = {
                **d,
                "disk": self.disk,
                "network": self.net_if_addrs,
                "users": self.users,
                "requirements": self.requirements,
            }
        return d

    def monitor(self):
        p = self.settings.x_sys_label
        d = {
            **{
                f"{p}cpu/pct/{i}": v
                for i, v in enumerate(psutil.cpu_percent(percpu=True))
            },
            **{
                f"{p}mem/{k}": v
                for k, v in psutil.virtual_memory()._asdict().items()
                if k in ("active", "used")
            },
            **{
                f"{p}disk/{k}": v
                for k, v in psutil.disk_usage(self.settings.work_dir())
                ._asdict()
                .items()
                if k in ("used")
            },
            **{
                f"{p}net/{k}": v
                for k, v in psutil.net_io_counters()._asdict().items()
                if k.startswith("bytes")
            },
        }
        if self.gpu:
            if self.gpu.get("nvidia"):
                import pynvml

                for h in self.gpu["nvidia"]["handles"]:
                    dev = (
                        str(pynvml.nvmlDeviceGetName(h))[2:-1].lower().replace(" ", "_")
                    )
                    d[f"{p}gpu/nvda/{dev}/pct"] = pynvml.nvmlDeviceGetUtilizationRates(
                        h
                    ).gpu
                    d[f"{p}gpu/nvda/{dev}/mem/pct"] = (
                        pynvml.nvmlDeviceGetUtilizationRates(h).memory
                    )
                    d[f"{p}gpu/nvda/{dev}/power"] = pynvml.nvmlDeviceGetPowerUsage(h)
        return d

    def monitor_human(self):
        d = {
            "cpu": {
                "percent": psutil.cpu_percent(percpu=True),
                "freq": [i.current for i in psutil.cpu_freq(percpu=True)],
            },
            "memory": {
                "virt": {
                    k: to_human(v)
                    for k, v in psutil.virtual_memory()._asdict().items()
                    if k != "percent"
                },
            },
            "disk": {
                "out": to_human(psutil.disk_io_counters().read_bytes),
                "in": to_human(psutil.disk_io_counters().write_bytes),
                "usage": {
                    k: to_human(v)
                    for k, v in psutil.disk_usage(self.settings.work_dir())
                    ._asdict()
                    .items()
                    if k != "percent"
                },
            },
            "network": {
                "out": to_human(psutil.net_io_counters().bytes_sent),
                "in": to_human(psutil.net_io_counters().bytes_recv),
            },
        }
        with self.proc.oneshot():  # perf
            d["process"] = {
                **self.proc.as_dict(
                    attrs=["status", "cpu_percent", "memory_percent", "num_threads"]
                ),
                "memory": to_human(self.proc.memory_info().rss),
            }
        if self.gpu:
            d["gpu"] = {}
            if self.gpu.get("nvidia"):
                import pynvml

                d["gpu"]["nvidia"] = {}
                d["gpu"]["nvidia"]["devices"] = [
                    {
                        "name": pynvml.nvmlDeviceGetName(h),
                        "temp": pynvml.nvmlDeviceGetTemperature(
                            h, pynvml.NVML_TEMPERATURE_GPU
                        ),
                        "gpu_percent": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                        "memory": {
                            "percent": pynvml.nvmlDeviceGetUtilizationRates(h).memory,
                            "used": to_human(pynvml.nvmlDeviceGetMemoryInfo(h).used),
                            "total": to_human(pynvml.nvmlDeviceGetMemoryInfo(h).total),
                        },
                        "power": {
                            "usage": pynvml.nvmlDeviceGetPowerUsage(h),
                            "limit": pynvml.nvmlDeviceGetEnforcedPowerLimit(h),
                        },
                        "pid": [
                            p.pid
                            for p in (
                                pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                                + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
                            )
                        ],
                    }
                    for h in self.gpu["nvidia"]["handles"]
                ]
        if self.settings.mode == "debug":
            d["memory"]["swap"] = {
                k: to_human(v)
                for k, v in psutil.swap_memory()._asdict().items()
                if k != "percent"
            }
        # exit(1)
        return d
