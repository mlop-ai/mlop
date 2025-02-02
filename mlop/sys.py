import platform

import psutil

from .util import to_human


class System:
    def __init__(self):
        self.uname = platform.uname()._asdict()

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

    def __getattr__(self, name):
        return self.get_psutil(name)

    def get_psutil(self, name):  # handling os specific methods
        if hasattr(psutil, name):
            return getattr(psutil, name)
        else:
            return None

    def info(self, debug=False):
        d = {
            "platform": self.uname,
            "cpu": {
                "physical": self.cpu_count(logical=False),
                "virtual": self.cpu_count(logical=True),
                "freq": {
                    "min": self.cpu_freq_min,
                    "max": self.cpu_freq_max,
                },
            },
            "memory": {
                "virtual": to_human(self.svmem["total"]),
                "swap": to_human(self.sswap["total"]),
            },
            "boot_time": self.boot_time,
        }
        if debug:
            d = {
                **d,
                "disk": self.disk,
                "network": self.net_if_addrs,
                "users": self.users,
            }
        return d
