"""Finding the cbc processes PuLP shells out to, by parent, from /proc.

Shared by the gunicorn master, which reaps solvers orphaned by a killed worker, and the worker,
which kills its own solver when CBC runs past its time limit. Standard library only, so the
master does not import the solver stack.
"""
import os


def solver_pids(parent, proc="/proc", name="cbc"):
    """PIDs of `name` processes whose parent is `parent`.

    Yields nothing without procfs, which is a development machine rather than the container.
    """
    try:
        entries = os.listdir(proc)
    except OSError:
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
                if f.read().rsplit(")", 1)[1].split()[1] != str(parent):
                    continue
        except (OSError, IndexError):
            # the process ended while being read, which is the outcome we wanted anyway
            continue
        yield int(entry)
