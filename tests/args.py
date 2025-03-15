import argparse
import importlib
import os

import keyring

AUTH = "mlpi_public_use_only_"
URL_LIST = {
    "REMOTE": "https://server.mlop.ai",
    "REMOTE_API": "https://api.mlop.ai/api/create-run",
}


def parse(TAG):
    parser = argparse.ArgumentParser(description=f"{TAG}: script for logger testing")
    parser.add_argument(
        "lib",
        choices=["m", "r", "w"],
        nargs="?",
        help="Library: mlop (local) [m], mlop (vps) [r], alternative [w]",
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

        d["x_log_level"] = 10
    else:
        print(f"{tag}: Using mlop")
        import mlop

        if args.lib == "m" or args.lib == "r":
            d["auth"] = AUTH
            if args.lib == "m":
                d["url"] = URL_LIST["LOCAL_RUST"]
            elif args.lib == "r":
                d["url"] = URL_LIST["REMOTE"]
            d["url_data"] = f"{d['url']}/ingest/metrics"
            d["url_file"] = f"{d['url']}/files"
            d["url_message"] = f"{d['url']}/ingest/logs"
            d["url_status"] = URL_LIST["REMOTE_API"]

        if args.debug == "db":
            d["disable_iface"] = True
        elif args.debug == "if":
            d["disable_store"] = True
        else:
            if tag == "METRIC":  # TODO: fix store implementation
                d["disable_store"] = True
                # d["x_log_level"] = 10

    if args.proxy == "e" or args.debug == "test":
        d["http_proxy"] = "http://localhost:8888"
        d["https_proxy"] = "http://localhost:8888"
        d["insecure_disable_ssl"] = True

    return mlop, d
