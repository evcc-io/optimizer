"""The hard limit on CBC: a cbc running past its -sec limit is killed by wall clock (#194).
The HiGHS side, an interrupt through its callback, is in test_highs.py."""
import os
import shutil
import subprocess
import time

import pytest
from test_gunicorn_conf import fake_proc
from test_objective_split import build

from optimizer.processes import solver_pids
from optimizer.settings import OptimizerSettings


def test_solvers_are_found_by_parent(tmp_path):
    fake_proc(tmp_path, 20, "gunicorn", 1)
    fake_proc(tmp_path, 30, "cbc", 20)   # this worker's solver
    fake_proc(tmp_path, 31, "cbc", 21)   # another worker's, must survive
    fake_proc(tmp_path, 32, "python3", 20)

    assert list(solver_pids(20, str(tmp_path))) == [30]


@pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs procfs')
def test_overrunning_solver_is_killed(tmp_path):
    # a stand in for cbc stuck in its root node: comm is taken from the name it is started by
    cbc = tmp_path / 'cbc'
    cbc.symlink_to(shutil.which('sleep'))
    optimizer = build('012-early-charging-not-perfect')
    optimizer.settings = OptimizerSettings(solver='cbc')

    started = time.monotonic()
    with optimizer._hard_limit(0.2) as killed:
        returncode = subprocess.Popen([str(cbc), '10']).wait()

    assert killed.is_set()
    assert returncode == -9
    assert time.monotonic() - started < 2


@pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs procfs')
def test_solver_finishing_in_time_is_left_alone(tmp_path):
    cbc = tmp_path / 'cbc'
    cbc.symlink_to(shutil.which('sleep'))
    optimizer = build('012-early-charging-not-perfect')
    optimizer.settings = OptimizerSettings(solver='cbc')

    with optimizer._hard_limit(1) as killed:
        returncode = subprocess.Popen([str(cbc), '0.1']).wait()
    # the timer is cancelled with the block, so it cannot fire on a later solve
    time.sleep(1.2)

    assert not killed.is_set()
    assert returncode == 0
