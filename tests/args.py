import argparse
import functools
import importlib
import os
import platform
import sys
import time
from typing import Any, Callable

import keyring

AUTH = "mlpi_public_use_only_"
URL_LIST = {
    "DEV": {
        "APP": "https://dev.mlop.ai",
        "API": "https://api-dev.mlop.ai",
        "INGEST": "https://ingest-dev.mlop.ai",
        "PY": "https://py-dev.mlop.ai",
    },
    "PROD": {
        "APP": "https://app.mlop.ai",
        "API": "https://api-prod.mlop.ai",
        "INGEST": "https://ingest-prod.mlop.ai",
        "PY": "https://py-prod.mlop.ai",
    },
}
PLATFORM_PREFS = {
    "Darwin": {
        # TODO: for server to catch up and sync up live, may potentially tell server to only respond a status code when the entire batch has been processed in the backend db
        "metric": {
            "NUM_EPOCHS": 50_000,
            "ITEM_PER_EPOCH": 20,
        },
        "image": {
            "NUM_EPOCHS": 10,
            "ITEM_PER_EPOCH": 10,
        },
        "audio": {
            "NUM_EPOCHS": 10,
            "ITEM_PER_EPOCH": 10,
        },
        "video": {
            "NUM_EPOCHS": 4,
            "ITEM_PER_EPOCH": 4,
        },
    },
    "Linux": {
        "metric": {
            "NUM_EPOCHS": 10,
            "ITEM_PER_EPOCH": 10,
        },
        "image": {
            "NUM_EPOCHS": 2,
            "ITEM_PER_EPOCH": 10,
        },
        "audio": {
            "NUM_EPOCHS": 2,
            "ITEM_PER_EPOCH": 10,
        },
        "video": {
            "NUM_EPOCHS": 2,
            "ITEM_PER_EPOCH": 10,
        },
    },
}


def get_prefs(TAG):
    return PLATFORM_PREFS[platform.system()][TAG]


def timer(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        s = time.time()
        print(f"timer: Started {func.__name__}")
        r = func(*args, **kwargs)
        e = time.time()
        print(f"timer: Finished {func.__name__}: took {e - s:.4f} seconds")
        return r

    return wrapper


def parse(TAG):
    parser = argparse.ArgumentParser(description=f"{TAG}: script for logger testing")
    parser.add_argument(
        "lib",
        choices=["d", "p", "l", "w"],
        nargs="?",
        help="Library: mlop (dev) [d], mlop (prod) [p], mlop (local) [l], alternative [w]",
    )
    parser.add_argument(
        "debug",
        choices=["test", "if", "db"],
        nargs="?",
        help="Debug: testing [test], upload-only [if], store-only [db]",
    )
    parser.add_argument(
        "proxy",
        choices=["e", "d"],
        nargs="?",
        help="Proxy: enabled [e] or disabled [d]",
    )
    return parser.parse_args()


def read_sets_compat(args, tag):
    os.makedirs(".mlop", exist_ok=True)
    d = {}

    if args.lib == "w":
        print(f"{tag}: Using alternative")
        try:
            module = keyring.get_password("mlop", "alternative")
        except Exception as e:
            pass
        if not module or module == "":
            module = input("Enter name: ")
        try:
            keyring.set_password("mlop", "alternative", module)
        except Exception as e:
            pass
        mlop = importlib.import_module(module)

        d["host"] = "localhost"
        d["x_log_level"] = 10
        d["disable_git"] = True
        d["save_code"] = False
        # d["mode"] = "offline"
    else:
        print(f"{tag}: Using mlop")
        import mlop

        if args.lib == "d" or args.lib == "l":
            # d["auth"] = AUTH
            if args.lib == "d":
                urls = URL_LIST["DEV"]

            d["_url"] = urls["APP"]
            d["_url_api"] = urls["API"]
            d["_url_py"] = urls["PY"]
            d["_url_ingest"] = urls["INGEST"]

            d["url_token"] = f"{d['_url']}/api-keys"
            d["url_trigger"] = f"{d['_url_py']}/api/runs/triggers"
            d["url_start"] = f"{d['_url_api']}/api/runs/create"
            d["url_stop"] = f"{d['_url_api']}/api/runs/status/update"
            d["url_meta"] = f"{d['_url_api']}/api/runs/logName/add"
            d["url_graph"] = f"{d['_url_api']}/api/runs/modelGraph/create"
            d["url_num"] = f"{d['_url_ingest']}/ingest/metrics"
            d["url_data"] = f"{d['_url_ingest']}/ingest/data"
            d["url_file"] = f"{d['_url_ingest']}/files"
            d["url_message"] = f"{d['_url_ingest']}/ingest/logs"
            d["url_view"] = None

        if args.debug == "db":
            d["disable_iface"] = True
        elif args.debug == "if":
            d["disable_store"] = True

        if tag == "metric":  # TODO: fix store implementation
            d["disable_store"] = True
            # d["x_log_level"] = 10

    if args.proxy == "e" or args.debug == "test":
        d["http_proxy"] = "http://localhost:8888"
        d["https_proxy"] = "http://localhost:8888"
        d["insecure_disable_ssl"] = True

    return mlop, d


@timer
def init_test(TAG):
    mlop, settings = read_sets_compat(parse(TAG), TAG)
    run = mlop.init(dir=".mlop/", project="test-" + TAG, settings=settings)
    sys.stderr.write("script: Started stderr logging\n")
    return mlop, run
