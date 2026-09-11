
from __future__ import annotations

import ctypes
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1


def die_with_parent() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):
        return

    if os.getppid() == 1:
        os._exit(1)
