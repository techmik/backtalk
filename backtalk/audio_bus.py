# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process-global coordination for a PortAudio reinitialize.

sd._terminate() / sd._initialize() rebuilds the device list for the WHOLE
process -- the only in-process way to notice the OS default output device
changed while backtalk was running, and it invalidates every open stream
(the mouth's output, the mic's input) across the call.

Why it's needed: unplugging Bluetooth earbuds paired THROUGH a USB dongle
(a Creative BT-W6) throws no PortAudio error -- the dongle is still a live
device as far as Windows and PortAudio's cached table are concerned, so
audio goes silently nowhere and mouth.py's error-driven _reinit_audio
never fires. The mouth instead watches for an idle stretch and asks for a
refresh here; the mic parks its stream while the flag is up, reopens after.
"""
import threading

import sounddevice as sd

from backtalk.vlog import log

_reinit_wanted = threading.Event()   # a refresh is pending; stream owners yield
_mic_parked = threading.Event()      # the open mic closed its stream, is waiting
_mic_listening = threading.Event()   # listen_once (hands-free) is running
_ptt_active = threading.Event()      # a hold-to-talk capture is live -- don't disturb
_lock = threading.Lock()


def request_reinit() -> None:
    _reinit_wanted.set()


def reinit_wanted() -> bool:
    return _reinit_wanted.is_set()


def park_mic() -> None:
    _mic_parked.set()


def unpark_mic() -> None:
    _mic_parked.clear()


def wait_mic_parked(timeout: float) -> bool:
    return _mic_parked.wait(timeout)


def mic_listen_start() -> None:
    _mic_listening.set()


def mic_listen_stop() -> None:
    _mic_listening.clear()
    _mic_parked.clear()


def mic_listening() -> bool:
    return _mic_listening.is_set()


def ptt_capture_start() -> None:
    _ptt_active.set()


def ptt_capture_end() -> None:
    _ptt_active.clear()


def ptt_active() -> bool:
    return _ptt_active.is_set()


def run_reinit() -> None:
    """Rebuild PortAudio's device list. Idempotent and lock-guarded, so
    whichever side gets here first does the work and the other is a no-op.
    Callers must have closed their own stream first."""
    with _lock:
        if not _reinit_wanted.is_set():
            return
        try:
            sd._terminate()
            sd._initialize()
            log("[audio] PortAudio device list rebuilt (idle output refresh)")
        except Exception as e:
            log(f"[audio] reinit failed: {e}")
        finally:
            _reinit_wanted.clear()
