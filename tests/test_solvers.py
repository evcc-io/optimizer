import os
import signal
import time

from optimizer.solvers import solver_wall, solvers_of


def fake_proc(root, pid, comm, ppid):
    entry = root / str(pid)
    entry.mkdir()
    (entry / "comm").write_text(comm + "\n")
    # the real format, comm in parens between pid and state
    (entry / "stat").write_text(f"{pid} ({comm}) R {ppid} {pid} 0 0 -1 4194304 500 0\n")


def test_only_the_named_parents_solvers_are_picked_up(tmp_path):
    fake_proc(tmp_path, 1, "gunicorn", 0)
    fake_proc(tmp_path, 20, "gunicorn", 1)        # a live worker
    fake_proc(tmp_path, 30, "cbc", 20)            # solving for that worker, must survive
    fake_proc(tmp_path, 40, "cbc", 1)             # orphan, its worker was killed
    fake_proc(tmp_path, 50, "python3", 1)         # reparented, but not a solver
    (tmp_path / "self").mkdir()                   # /proc holds non numeric entries too

    assert list(solvers_of(1, str(tmp_path))) == [40]
    assert list(solvers_of(20, str(tmp_path))) == [30]


def test_a_process_ending_mid_read_is_skipped(tmp_path):
    fake_proc(tmp_path, 40, "cbc", 1)
    (tmp_path / "41").mkdir()  # exited between listdir and open, so it has no comm

    assert list(solvers_of(1, str(tmp_path))) == [40]


def test_a_solve_past_the_wall_loses_its_solver(tmp_path):
    fake_proc(tmp_path, 40, "cbc", os.getpid())   # this process's own solver
    fake_proc(tmp_path, 41, "cbc", 1)             # somebody else's, not ours to kill
    killed = []

    with solver_wall(0.05, str(tmp_path), lambda pid, sig: killed.append((pid, sig))):
        time.sleep(0.2)

    assert killed == [(40, signal.SIGKILL)]


def test_a_solve_inside_the_wall_keeps_its_solver(tmp_path):
    fake_proc(tmp_path, 40, "cbc", os.getpid())
    killed = []

    with solver_wall(5, str(tmp_path), lambda pid, sig: killed.append((pid, sig))):
        pass
    time.sleep(0.1)  # the timer would have to have been cancelled, not merely be pending

    assert killed == []


def test_no_limit_means_no_wall(tmp_path):
    fake_proc(tmp_path, 40, "cbc", os.getpid())
    killed = []

    with solver_wall(None, str(tmp_path), lambda pid, sig: killed.append((pid, sig))):
        time.sleep(0.1)

    assert killed == []
