"""Test-wide isolation from the LIVE signal bus.

backtalk.signals resolves every bus path once, at import, from signals_dir
in backtalk.json -- the same folder a running voice session and its face
are using. Without this, a test that drives the brain through a degraded
failover writes "Claude unreachable" into the live chat transcript and a
fake degraded flag into .voice_session (found 2026-09-22: 13 phantom lines
in a live session's chat box). Every bus path, plus brain's resume file,
is repointed at a throwaway folder for the whole run. vlog.LOG_PATH goes
there too, so test runs stop appending fake lines to logs/backtalk.log
(the receipts file for real voice sessions).
"""
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_signal_bus(tmp_path_factory):
    from backtalk import brain, signals, vlog

    bus = str(tmp_path_factory.mktemp("signal_bus"))
    live = signals._DIR
    saved = {}
    for name in dir(signals):
        val = getattr(signals, name)
        if (name.startswith("_") and name.endswith("_FILE")
                and isinstance(val, str) and val.startswith(live)):
            saved[name] = val
            setattr(signals, name, os.path.join(bus, os.path.basename(val)))
    # the barehands mirror writes outside signals_dir entirely: switch it off
    for name in ("_BH_STATE", "_BH_WAVE"):
        saved[name] = getattr(signals, name)
        setattr(signals, name, "")
    saved_session_file = brain.SESSION_FILE
    brain.SESSION_FILE = os.path.join(bus, ".backtalk_session")
    saved_log_path = vlog.LOG_PATH
    vlog.LOG_PATH = Path(bus) / "backtalk.log"
    yield bus
    for name, val in saved.items():
        setattr(signals, name, val)
    brain.SESSION_FILE = saved_session_file
    vlog.LOG_PATH = saved_log_path
