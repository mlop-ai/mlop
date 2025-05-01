import copy
import importlib
import json
import logging
import os
import random
import shutil
import string
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, Sequence, Union

from .sets import get_console

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = "Util"


class ANSI:
    base = "\033["  # "\x1b["
    reset = f"{base}0m"
    bold = f"{base}1m"
    faint = f"{base}2m"
    italic = f"{base}3m"
    underline = f"{base}4m"
    slow_blink = f"{base}5m"
    rapid_blink = f"{base}6m"

    black = f"{base}30m"
    red = f"{base}31m"
    green = f"{base}32m"
    yellow = f"{base}33m"
    blue = f"{base}34m"
    purple = f"{base}35m"
    cyan = f"{base}36m"
    white = f"{base}37m"

    if not __import__("sys").stdout.isatty() and get_console() == "python":
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        if __import__("platform").system() == "Windows":
            os.system("")


def print_url(url):
    return f"{ANSI.underline}{url}{ANSI.reset}"


def find_node(nodes, id, key="nodes"):
    if nodes.get("id") == id:
        return nodes

    if key in nodes:
        for child in nodes[key]:
            result = find_node(child, id, key)
            if result is not None:
                return result

    return None


def import_lib(m, a="None"):
    try:
        return sys.modules[m]
    except KeyError:
        try:
            module = importlib.import_module(m)
            if a:
                globals()[a] = module
            return module
        except ImportError:
            logger.error(
                f"{tag}: {m} not installed; module-related functionality will be disabled"
            )
            return None


def update_node(src, dst):
    d = find_node(src, id=dst["id"], key="nodes")
    if d is not None:
        d = copy.deepcopy(d)
        if d.get("nodes"):
            del d["nodes"]
        dst.update(d)

    if dst.get("nodes"):
        for child in dst["nodes"]:
            child = update_node(src, child)

    return dst


def run_cmd(cmd="ls", timeout=10):
    if not shutil.which(cmd.split()[0]):
        return None

    try:
        r = subprocess.run(
            cmd.split(), check=False, capture_output=True, text=True, timeout=timeout
        )
        if r.returncode == 0:
            return r.stdout
        else:
            return r.stderr
    except FileNotFoundError:
        return None
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logger.error("%s: %s", tag, e)
        return None


def gen_id(seed=None, length=8) -> str:
    random.seed(uuid.uuid4().hex) if seed is None else random.seed(
        seed + uuid.uuid4().hex
    )
    base = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return "".join(random.choice(base) for _ in range(length))


def gen_ulid(base="0123456789ABCDEFGHJKMNPQRSTVWXYZ") -> str:  # py-ulid
    ulid = (int(time.time() * 1000) << 80) | random.getrandbits(80)

    encoded = []
    while ulid > 0:
        ulid, remainder = divmod(ulid, 32)
        encoded.append(base[remainder])
    return "".join(encoded[::-1]).rjust(26, base[0])


def to_dict(obj):
    attrs = {}
    for name in dir(obj):
        if not name.startswith("__") and not callable(getattr(obj, name)):
            attrs[name] = getattr(obj, name)
    return attrs


def to_json(data, file):
    if os.path.exists(file):
        with open(file, "r+") as f:
            try:
                content = json.load(f)
                if not isinstance(content, list):
                    logger.error(f"{tag}: file content must be a json list")
                    return
            except json.JSONDecodeError:
                logger.error(f"{tag}: file is not in json format")
                return
            content.extend(data)
            f.seek(0)
            json.dump(content, f, indent=4)
            f.truncate()
    else:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)


def to_human(n):
    symbols = ("K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return "%.1f%s" % (value, s)
    return "%sB" % n


def clean_dict(d):
    c = {}
    for k, v in d.items():
        if (
            k.startswith("_")
            or hasattr(v, "__dict__")
            or hasattr(v, "__slots__")
            or v is None
        ):
            continue
        c[k] = v
    return c


def dict_to_json(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in list(data):  # avoid RuntimeError if dict size changes
        val = data[key]
        if isinstance(val, dict):
            data[key] = dict_to_json(val)
        else:
            data[key] = val_to_json(val)
    return data


def val_to_json(val: Any) -> Union[Sequence, Dict, Any]:
    if isinstance(val, (int, float, str, bool)):
        return val
    elif isinstance(val, (list, tuple, range)):
        raise NotImplementedError()  # TODO: for files


def get_class(val: any) -> str:
    module_class = val.__class__.__module__ + "." + val.__class__.__name__
    return (
        val.__name__
        if module_class in ["builtins.module", "__builtin__.module"]
        else module_class
    )
