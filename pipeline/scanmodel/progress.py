"""Run a long external program with a progress bar (step / total, elapsed, ETA) and a clean log.

The program runs in a pseudo-terminal: Brush (Rust, `indicatif`) only draws its progress when it talks
to a terminal, and Python tools flush line by line there. Progress is read from what they print:
msplat's `step=  1200` lines, or an `indicatif` bar's `1200/7000`. Everything else is shown above the
bar and written to the log without terminal escape codes or redraws.
"""

import fcntl
import os
import pty
import re
import struct
import subprocess
import termios
from collections.abc import Callable
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .cache import console

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]")
STEP = re.compile(r"step=\s*(\d+)")
OF_TOTAL = re.compile(r"(\d+)\s*/\s*(\d+)")


def step_of(text: str, total: int) -> int | None:
    """The step count in a line of trainer output, if it has one."""
    m = STEP.search(text)
    if m:
        return int(m[1])
    for a, c in OF_TOTAL.findall(text):  # an indicatif bar: "1200/7000"
        if int(c) == total:
            return int(a)
    return None


def bar(unit: str) -> Progress:
    """A progress bar: the task's description, the bar, done/total `unit`, elapsed, ETA."""
    columns = (
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(unit),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    )
    return Progress(*columns, console=console, transient=False)


def run(
    cmd: list[str],
    cwd: Path,
    log: Path,
    total: int,
    label: str,
    show: Callable[[str], bool] = lambda line: True,
) -> int:
    """Run `cmd` with a progress bar towards `total` steps; returns its exit code. Lines that carry a
    step only move the bar; other lines go to the terminal if `show(line)` and always to `log`."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TERM": "xterm-256color"}
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))  # indicatif needs a width
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, env=env)
    os.close(slave)
    pending = ""
    # Line-buffered: the log follows the run (`tail -f`); a few lines a second cost nothing.
    with log.open("w", buffering=1) as f, bar("steps") as steps:
        task = steps.add_task(label, total=total)
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:  # the child closed the terminal
                break
            if not chunk:
                break
            pending += ANSI.sub("", chunk.decode("utf-8", "replace")).replace("\r\n", "\n")
            *done, pending = pending.split("\n")
            for line in done:
                *redraws, final = line.split("\r")  # a bar redrawn in place: only the last state is a line
                for text in [*redraws, final]:
                    step = step_of(text, total)
                    if step is not None:
                        steps.update(task, completed=min(step, total))
                if final.strip() and step_of(final, total) is None:
                    f.write(final + "\n")
                    if show(final):
                        steps.console.print(final, markup=False, highlight=False, soft_wrap=True)
                elif step_of(final, total) is not None and STEP.search(final):
                    f.write(final + "\n")  # msplat's step lines stay in the log (metrics parse it)
            redraws = pending.split("\r")  # a bar still being redrawn on the current line
            for text in redraws:
                step = step_of(text, total)
                if step is not None:
                    steps.update(task, completed=min(step, total))
            pending = redraws[-1]  # keep only its latest state
        if pending.strip():
            f.write(pending + "\n")
    os.close(master)
    return p.wait()
