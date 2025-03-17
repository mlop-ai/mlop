import getpass
import logging
import sys
import webbrowser

import httpx
import keyring
from keyrings.alt.file import PlaintextKeyring

from .log import setup_logger, teardown_logger
from .sets import Settings, get_console
from .util import ANSI, print_url

tlogger = logging.getLogger("auth")
tag = "Auth"


def login(settings=None, retry=False):
    settings = settings or Settings()
    setup_logger(settings=settings, logger=tlogger)
    try:
        auth = keyring.get_password(f"{settings.tag}", f"{settings.tag}")
    except keyring.errors.NoKeyringError:  # fallback
        keyring.set_keyring(PlaintextKeyring())
        auth = keyring.get_password(f"{settings.tag}", f"{settings.tag}")
    if settings.auth is None:
        if auth == "":
            keyring.delete_password(f"{settings.tag}", f"{settings.tag}")
        elif auth is not None:
            settings.auth = auth
    if settings.auth == "":
        tlogger.critical(
            "%s: authentication failed: the provided token cannot be empty", tag
        )
        settings.auth = "_key"
    client = httpx.Client(
        verify=True if not settings.insecure_disable_ssl else False,
        proxy=settings.http_proxy or settings.https_proxy or None
    )
    r = client.post(
        url=settings.url_login,
        headers={
            "Authorization": f"Bearer {settings.auth}",
        },
    )
    try:
        tlogger.info(f"{tag}: logged in as {r.json()['organization']['slug']}")
        keyring.set_password(f"{settings.tag}", f"{settings.tag}", f"{settings.auth}")
        teardown_logger(tlogger)
    except Exception as e:
        if retry:
            tlogger.warning("%s: authentication failed", tag)
        hint1 = f"{ANSI.cyan}- Please copy the API key provided in the web portal and paste it below"
        hint2 = f"- You can alternatively manually open {print_url(settings.url_token)}"
        hint3 = f"{ANSI.green}- You may exit at any time by pressing CTRL+C / ⌃+C"
        tlogger.info(
            f"{tag}: initializing authentication\n\n {hint1}\n\n {hint2}\n\n {hint3}\n"
        )
        webbrowser.open(url=settings.url_token)
        if get_console() == "jupyter":
            settings.auth = getpass.getpass(prompt="Enter API key: ")
        else:
            settings.auth = input(f"{ANSI.yellow}Enter API key: ")
        try:
            keyring.set_password(
                f"{settings.tag}", f"{settings.tag}", f"{settings.auth}"
            )
        except Exception as e:
            tlogger.critical(
                "%s: failed to save key to system keyring service: %s", tag, e
            )
        teardown_logger(tlogger)
        login(retry=True)


def logout(settings=None):
    settings = settings or Settings()
    setup_logger(settings=settings, logger=tlogger)
    try:
        keyring.delete_password(f"{settings.tag}", f"{settings.tag}")
    except keyring.errors.NoKeyringError:
        keyring.set_keyring(PlaintextKeyring())
        keyring.delete_password(f"{settings.tag}", f"{settings.tag}")
    except Exception as e:
        tlogger.warning("%s: failed to delete key from system keyring service: %s", tag, e)
    tlogger.info(f"{tag}: logged out")
    teardown_logger(tlogger)
