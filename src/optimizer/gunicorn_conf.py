"""gunicorn hooks. Wired with --config python:optimizer.gunicorn_conf.

A worker killed by --timeout dies by SIGKILL, so nothing in the worker runs on the way
out. PuLP solves by shelling out to cbc, and that child survives its parent: the -sec
limit stops belonging to anything once the parent is gone, so the orphan keeps a core
busy for as long as the replica lives. Six of eleven replicas were carrying one on
2026-07-25, two of them with more than five core hours on the clock each.

child_exit runs in the master, which is the one hook a SIGKILLed worker still triggers.
"""
import os
import signal


def orphaned_solvers(proc="/proc", name="cbc"):
    """PIDs of `name` processes that have been reparented to PID 1.

    The master runs as PID 1 in the container, so a solver whose worker is gone lands on
    it. A solver still owned by a live worker names that worker as its parent, never 1,
    which is what keeps this from killing a running solve.
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
                if f.read().strip() != name:
                    continue
            with open(os.path.join(proc, entry, "stat")) as f:
                # comm sits in parens and may hold spaces, so split off the last ')' first.
                # The fields after it are state, ppid, ...
                if f.read().rsplit(")", 1)[1].split()[1] != "1":
                    continue
        except (OSError, IndexError):
            # the process ended while being read, which is the outcome we wanted anyway
            continue
        yield int(entry)


def child_exit(server, worker):
    for pid in orphaned_solvers():
        server.log.warning("reaping orphaned solver %s left by worker %s", pid, worker.pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            server.log.warning("could not reap solver %s: %s", pid, e)
