# -*- coding: utf-8 -*-

import os
import time
from tqdm import tqdm

# ---------------- clean live progress ----------------
_PROGRESS_TTY = None
_PROGRESS_TTY_CHECKED = False


def _get_progress_tty():
    """Return controlling terminal for live tqdm; never send refreshes into log."""
    global _PROGRESS_TTY, _PROGRESS_TTY_CHECKED
    if _PROGRESS_TTY_CHECKED:
        return _PROGRESS_TTY

    _PROGRESS_TTY_CHECKED = True
    enabled = str(os.environ.get("CLEAN_TQDM_LIVE", "1")).strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    try:
        tty = open("/dev/tty", "w", buffering=1, encoding="utf-8", errors="replace")
        if tty.isatty():
            _PROGRESS_TTY = tty
            return tty
        tty.close()
    except Exception:
        pass
    return None


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def clean_progress(iterable, desc: str, disable: bool = False):
    """Terminal: one live tqdm line. Log: one completion summary line."""
    try:
        total = len(iterable)
    except Exception:
        total = None

    tty = None if disable else _get_progress_tty()
    start = time.perf_counter()
    completed = 0
    finished = False

    if tty is not None:
        bar = tqdm(
            iterable,
            desc=desc,
            total=total,
            dynamic_ncols=True,
            disable=False,
            leave=False,
            file=tty,
            mininterval=0.5,
        )
        try:
            for item in bar:
                completed += 1
                yield item
            finished = True
        finally:
            bar.close()
    else:
        for item in iterable:
            completed += 1
            yield item
        finished = True

    if finished:
        elapsed = max(time.perf_counter() - start, 1e-9)
        total_text = str(total) if total is not None else "?"
        print(
            f"[ProgressDone] {desc}: {completed}/{total_text} "
            f"elapsed={_format_elapsed(elapsed)} rate={completed / elapsed:.2f}it/s",
            flush=True,
        )

