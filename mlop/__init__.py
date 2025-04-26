from .auth import login, logout
from .data import Data, Graph, Histogram
from .file import Audio, File, Image, Video
from .init import finish, init
from .sets import Settings
from .sys import System

# TODO: setup preinit

_hooks = []
ops, log, watch, alert = None, None, None, None

__all__ = (
    "Data",
    "Graph",
    "Histogram",
    "File",
    "Image",
    "Audio",
    "Video",
    "System",
    "Settings",
    "alert",
    "init",
    "login",
    "logout",
    "watch",
    "finish",
)

__version__ = "0.0.0"
