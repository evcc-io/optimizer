from optimizer.gunicorn_conf import orphaned_solvers


def fake_proc(root, pid, comm, ppid):
    entry = root / str(pid)
    entry.mkdir()
    (entry / "comm").write_text(comm + "\n")
    # the real format, comm in parens between pid and state
    (entry / "stat").write_text(f"{pid} ({comm}) R {ppid} {pid} 0 0 -1 4194304 500 0\n")


def test_only_reparented_solvers_are_picked_up(tmp_path):
    fake_proc(tmp_path, 1, "gunicorn", 0)
    fake_proc(tmp_path, 20, "gunicorn", 1)        # a live worker
    fake_proc(tmp_path, 30, "cbc", 20)            # solving for that worker, must survive
    fake_proc(tmp_path, 40, "cbc", 1)             # orphan, its worker was killed
    fake_proc(tmp_path, 50, "python3", 1)         # reparented, but not a solver
    (tmp_path / "self").mkdir()                   # /proc holds non numeric entries too

    assert list(orphaned_solvers(str(tmp_path))) == [40]


def test_a_process_ending_mid_read_is_skipped(tmp_path):
    fake_proc(tmp_path, 40, "cbc", 1)
    (tmp_path / "41").mkdir()  # exited between listdir and open, so it has no comm

    assert list(orphaned_solvers(str(tmp_path))) == [40]
