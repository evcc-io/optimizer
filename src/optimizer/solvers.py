"""The cbc processes a worker owns, and the wall clock they are held to.

Two callers share the one /proc scan. The gunicorn master reaps solvers that were
reparented to it when it killed the worker they belonged to; a worker kills its own
solver when a solve outlives the wall below.

CBC needs the wall because -sec is not one. PuLP does pass it in elapsed time, but CBC
only tests it between branch and bound nodes, so a solve that grinds in presolve or the
root LP runs unbounded. Production saw that as a worker wedged past gunicorn's timeout
and SIGKILLed, taking the replica's queued requests with it: eight times in the ten
hours after 2026-07-26 11:00 UTC, each leaving an orphaned solver behind.
"""
import os
import signal
import threading
from contextlib import contextmanager


def solvers_of(ppid, proc="/proc"):
    """PIDs of the cbc processes whose parent is `ppid`.

    The master runs as PID 1 in the container, so a solver whose worker is gone lands on
    it. A solver still owned by a live worker names that worker as its parent, which is
    what keeps `solvers_of(1)` from killing a running solve.
    """
    try:
        entries = os.listdir(proc)
    except OSError:
        # no procfs, which is a development machine rather than the container
        return
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc, entry, "comm")) as f:
                if f.read().strip() != "cbc":
                    continue
            with open(os.path.join(proc, entry, "stat")) as f:
                # comm sits in parens and may hold spaces, so split off the last ')' first.
                # The fields after it are state, ppid, ...
                if f.read().rsplit(")", 1)[1].split()[1] != str(ppid):
                    continue
        except (OSError, IndexError):
            # the process ended while being read, which is the outcome we wanted anyway
            continue
        yield int(entry)


@contextmanager
def solver_wall(seconds, proc="/proc", kill=os.kill):
    """Kill this process's solvers if the block outlives `seconds`. No seconds, no wall.

    PuLP raises when its solver dies, so the request fails at the wall instead of holding
    the worker until gunicorn kills that too.
    """
    if not seconds:
        yield
        return

    def fire():
        for pid in solvers_of(os.getpid(), proc):
            try:
                kill(pid, signal.SIGKILL)
            except OSError:
                # it finished on its own between the scan and the signal
                pass

    timer = threading.Timer(seconds, fire)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
