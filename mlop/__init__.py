from .auth import login, logout
from .compat.torch import watch
from .data import Data, Graph, Histogram
from .file import Audio, File, Image
from .init import init
from .sets import Settings, setup
from .sys import System

# TODO: setup preinit

ops = None
_hooks = []

__all__ = (
    "Data",
    "Graph",
    "Histogram",
    "File",
    "Image",
    "Audio",
    "System",
    "Settings",
    "init",
    "login",
    "logout",
    "setup",
    "watch"
)

__version__ = "0.0.0"
