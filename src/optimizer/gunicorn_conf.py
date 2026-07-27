"""gunicorn hooks. Wired with --config python:optimizer.gunicorn_conf.

A worker killed by --timeout dies by SIGKILL, so nothing in the worker runs on the way
out, and the solver it left keeps a core busy for as long as the replica lives. Six of
eleven replicas were carrying one on 2026-07-25, two of them with more than five core
hours on the clock each. optimizer.solvers has the rest of that story.

child_exit runs in the master, which is the one hook a SIGKILLed worker still triggers.
The master then reaps the solver it adopted, and logs that as a worker sent SIGKILL —
that line names the solver's pid, not a second worker, and has nothing to do with memory.
"""
import os
import signal

from optimizer.solvers import solvers_of


def child_exit(server, worker):
    for pid in solvers_of(os.getpid()):
        server.log.warning("reaping orphaned solver %s left by worker %s", pid, worker.pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            server.log.warning("could not reap solver %s: %s", pid, e)
