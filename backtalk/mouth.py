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
"""The mouth — streaming sentence-chunked TTS, played through one
long-lived output stream.

Default engine: Kokoro, in-process. Local, free, no server, no API key,
~0.2s to first audio once warm. Optional premium engine: ElevenLabs on
YOUR key — read from the system keychain, never from a file (see
_get_elevenlabs_key) — with Kokoro as the automatic fallback: the voice
degrades instead of going mute if the cloud fails.

Sentences are synthesized one at a time and queued for playback, so the
first sentence is audible while later ones are still rendering. Playback
is cancellable mid-word: set the stop event and the speaker goes silent
within one audio block plus the device buffer (~0.15s).

HARD-WON AUDIO LAW #1 — ONE long-lived OutputStream, reused for every
sentence for the life of the process. A fresh stream per sentence gives
an audible onset blip or a beat of dead air on plenty of audio setups
(USB interfaces, Bluetooth, streaming mixers that latch onto each new
stream late). Proven by A/B test; do not "simplify" this away.

HARD-WON AUDIO LAW #2 — buffer ~0.75s of synthesized audio before a
sentence starts playing, so a slower machine never underruns into
slow-motion garble.
"""
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd

from backtalk.config import CFG
from backtalk.vlog import log

KOKORO_RATE = 24000
EL_RATE = 44100
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_pipe = None
_pipe_lock = threading.Lock()


def _ensure_espeak():
    """kokoro phonemizes through system espeak-ng (its bundled loader
    ships a broken build path — found the hard way; upstream's own docs
    say install the system package). Help phonemizer find it in the
    usual homes when the env isn't already set."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = (
        "/opt/homebrew/lib/libespeak-ng.dylib",       # macOS arm64 (brew)
        "/usr/local/lib/libespeak-ng.dylib",          # macOS intel (brew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # debian/ubuntu
        "/usr/lib/libespeak-ng.so.1",                 # other linux
        "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",       # windows
        "C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
    )
    for lib in candidates:
        if os.path.exists(lib):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
            break


# Every espeak library filename phonemizer might copy, on any platform. A
# directory holding exactly one of these and nothing else is a phonemizer
# scratch dir and is not plausibly anything else.
_ESPEAK_LIB_NAMES = (
    "espeak-ng.dll",
    "libespeak-ng.dll",
    "libespeak-ng.so",
    "libespeak-ng.so.1",
    "libespeak-ng.dylib",
)


def _is_orphan_espeak_tempdir(path: str) -> bool:
    """True only for a directory whose ENTIRE contents are one espeak
    library. That signature is what makes it safe to point a delete at a
    shared temp folder: one file, and its name is one of five."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return len(entries) == 1 and entries[0] in _ESPEAK_LIB_NAMES


def _sweep_orphan_espeak_tempdirs():
    """Delete espeak scratch dirs left behind by previous runs.

    phonemizer copies the espeak shared library into a fresh temp dir for
    every backend it builds, because espeak-ng keeps its state in globals
    and the loader refuses the same file twice. Kokoro builds several
    backends, so ONE start leaves several behind.

    On POSIX that cleanup rides a finalizer and usually happens. On
    Windows phonemizer can only register it with atexit, and atexit does
    not run when a process is KILLED rather than exited -- so anything
    stopping the voice line by terminating it, which is most launchers and
    every supervisor, leaks every directory it ever made. Sixty had piled
    up on the machine where this was found, and fifteen were sitting on
    the author's own Mac when it was reviewed: the POSIX path is not as
    reliable as it looks either. The count only ever grows.

    Patching phonemizer where it is installed is not a fix, because the
    launcher runs a dependency sync that would overwrite it. Sweeping at
    our own startup bounds the total at one run's worth instead.

    Two things make deleting from a shared temp folder safe, and only the
    first is ours: the signature above is narrow enough that nothing else
    matches it, and anything we are not permitted to remove raises and is
    skipped. On Windows a loaded library cannot be deleted at all, so a
    live instance is protected by the OS rather than by us noticing it.
    POSIX does not work that way, but a process that has already mapped
    the library keeps it after the unlink, so a running instance is
    unharmed either way.
    """
    root = tempfile.gettempdir()
    swept = 0
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _is_orphan_espeak_tempdir(path):
            continue
        try:
            shutil.rmtree(path)
            swept += 1
        except OSError:
            pass          # in use, or not ours. Leaving it is correct.
    if swept:
        log(f"[mouth] swept {swept} orphaned espeak temp dir(s)")


def warm():
    """Load the Kokoro pipeline (first call downloads the model to the
    HF cache). Called at startup while the greeting text is composed."""
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            _ensure_espeak()
            # Before kokoro makes this run's scratch dirs, clear the ones
            # earlier runs could not clean up on their way out.
            _sweep_orphan_espeak_tempdirs()
            from kokoro import KPipeline
            # The voice name's first letter IS the language pipeline:
            # a=American English, b=British English, e/f/h/i/j/p/z = the
            # other shipped languages. bm_lewis -> 'b'.
            lang = (CFG["voice"] or "bm_lewis")[0]
            log(f"[mouth] loading kokoro (lang '{lang}', "
                f"voice {CFG['voice']})...")
            _pipe = KPipeline(lang_code=lang)
            log("[mouth] voice ready")
    return _pipe


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _stream_kokoro(text: str):
    """One sentence -> int16 PCM chunks at 24kHz, in-process."""
    pipe = warm()
    try:
        speed = float(CFG.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    for _, _, audio in pipe(text, voice=CFG["voice"], speed=speed):
        a = np.asarray(audio, dtype=np.float32)
        if a.size:
            yield (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)


def _stream_elevenlabs(text: str, timeout: float):
    """ElevenLabs -> ffmpeg streaming decode -> int16 PCM at 44.1kHz.

    THE ELEVENLABS DOCTRINE, learned the expensive way:
    - fetch mp3_44100_128 and decode locally (raw 44.1k PCM needs their
      Pro tier; the mp3 decode hides inside network wait anyway)
    - turbo model, stability from config (0.5 default), similarity 0.75
    - never the multilingual model for English, never style above 0 —
      both make delivery slow and dull
    - their site previews are MASTERED demo clips; raw API output never
      matches them, so master locally (the ffmpeg chain in config)
    ffmpeg reads stdin as we feed it, so playback still starts before
    synthesis finishes."""
    import subprocess

    import httpx

    el = CFG["elevenlabs"]
    key = _get_elevenlabs_key()
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/"
           f"{el['voice_id']}/stream?output_format=mp3_44100_128")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-af", el["master"],
         "-f", "s16le", "-ar", str(EL_RATE), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    feed_error: list = []

    def _feed():
        try:
            with httpx.stream("POST", url, headers={"xi-api-key": key},
                              json={"text": text, "model_id": el["model"],
                                    "voice_settings": {
                                        "stability": el.get("stability", 0.5),
                                        "similarity_boost": 0.75}},
                              timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    proc.stdin.write(chunk)
        except Exception as e:
            feed_error.append(e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    carry = b""
    got_audio = False
    while True:
        data = proc.stdout.read(8820)
        if not data:
            break
        data = carry + data
        usable = len(data) - (len(data) % 2)
        carry = data[usable:]
        if usable:
            got_audio = True
            yield np.frombuffer(data[:usable], dtype=np.int16)
    proc.wait(timeout=10)
    if feed_error and not got_audio:
        raise feed_error[0]


def _stream_elevenlabs_v3(text: str, timeout: float):
    """Eleven v3 Conversational via the Text to Dialogue REST stream ->
    ffmpeg streaming decode -> int16 PCM at 44.1kHz.

    Same transport as _stream_elevenlabs (httpx stream -> the ffmpeg
    master chain -> PCM), but the v3 dialogue models are NOT served from
    /text-to-speech, so the endpoint and body differ: inputs[] with a
    per-line voice_id, model_id, and settings.stability only (no
    similarity_boost, no style). synth_stream picks this when
    elevenlabs.model starts with "eleven_v3". Kept as its own function so
    the turbo path stays byte-for-byte untouched and this trial reverts
    cleanly. A request caps at 2000 characters across all inputs; an
    over-long chunk raises here and synth_stream degrades it to Kokoro."""
    import subprocess

    import httpx

    if len(text) > 2000:
        raise ValueError("dialogue request over the 2000-char limit")

    el = CFG["elevenlabs"]
    key = _get_elevenlabs_key()
    url = ("https://api.elevenlabs.io/v1/text-to-dialogue/stream"
           "?output_format=mp3_44100_128")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-af", el["master"],
         "-f", "s16le", "-ar", str(EL_RATE), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    feed_error: list = []

    def _feed():
        try:
            with httpx.stream("POST", url, headers={"xi-api-key": key},
                              json={"inputs": [{"text": text,
                                                "voice_id": el["voice_id"]}],
                                    "model_id": el["model"],
                                    "settings": {
                                        "stability": el.get("stability", 0.5)}},
                              timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=4096):
                    proc.stdin.write(chunk)
        except Exception as e:
            feed_error.append(e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    carry = b""
    got_audio = False
    while True:
        data = proc.stdout.read(8820)
        if not data:
            break
        data = carry + data
        usable = len(data) - (len(data) % 2)
        carry = data[usable:]
        if usable:
            got_audio = True
            yield np.frombuffer(data[:usable], dtype=np.int16)
    proc.wait(timeout=10)
    if feed_error and not got_audio:
        raise feed_error[0]


_el_key_cache: str | None = None


def _key_slot() -> str:
    """The credential-store entry name, so someone who already keeps a key
    under their own name points at it instead of storing a second copy."""
    return str(CFG["elevenlabs"].get("key_slot") or "backtalk-elevenlabs")


def _get_elevenlabs_key() -> str:
    """The API key, from the most secure store available — NEVER from a
    file in this repo. Lookup order:
      1. macOS Keychain, item `backtalk-elevenlabs` by default (change it
         with elevenlabs.key_slot) — seed it once with:
         security add-generic-password -a "$USER" -s backtalk-elevenlabs -T /usr/bin/security -w
         (it prompts for the secret; -T lets this code read it without a
         GUI prompt every launch)
      2. Linux secret-tool (libsecret):
         secret-tool store --label backtalk service backtalk-elevenlabs
      3. the ELEVENLABS_API_KEY environment variable — the last-resort
         fallback, and the only option on Windows for now. Know the
         tradeoff: an export line in a shell profile is a plaintext key
         on disk, which is exactly what the keychain path avoids."""
    global _el_key_cache
    if _el_key_cache is not None:
        return _el_key_cache
    import subprocess
    key = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["security", "find-generic-password",
                                "-s", _key_slot(), "-w"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                key = r.stdout.strip()
        elif sys.platform.startswith("linux"):
            from shutil import which
            if which("secret-tool"):
                r = subprocess.run(["secret-tool", "lookup", "service",
                                    _key_slot()],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    key = r.stdout.strip()
    except Exception:
        pass
    _el_key_cache = key or os.environ.get("ELEVENLABS_API_KEY", "")
    return _el_key_cache


def _elevenlabs_ready() -> bool:
    el = CFG["elevenlabs"]
    return bool(el.get("enabled") and el.get("voice_id")
                and _get_elevenlabs_key())


def synth_stream(text: str, timeout: float = 30.0):
    """One sentence -> yields (sample_rate, pcm_chunk) as the TTS
    renders. ElevenLabs when configured, Kokoro otherwise — and Kokoro
    as the fallback on ANY ElevenLabs failure. Degrade, never mute."""
    if _elevenlabs_ready():
        import time
        el = CFG["elevenlabs"]
        v3 = str(el.get("model", "")).startswith("eleven_v3")
        stream_fn = _stream_elevenlabs_v3 if v3 else _stream_elevenlabs
        try:
            t0 = time.perf_counter()
            first = True
            for pcm in stream_fn(text, timeout):
                if first:
                    log(f"[mouth] ttfa {'v3' if v3 else 'turbo'} "
                        f"{(time.perf_counter() - t0) * 1000:.0f}ms")
                    first = False
                yield EL_RATE, pcm
            return
        except Exception as e:
            log(f"[mouth] elevenlabs failed ({str(e)[:60]}) — "
                f"falling back to {CFG['voice']}")
    for pcm in _stream_kokoro(text):
        yield KOKORO_RATE, pcm


_speaker_device_warned = False


def _speaker_index():
    """Resolve speaker_device (a device NAME) to a CONCRETE output
    index. The OUTPUT twin of ears._mic_index().

    Returns a concrete index whenever a default exists — never None in
    that case — because a stored None is exactly what hides a device
    renumbering: a phone or a Bluetooth dongle joining or leaving
    shifts PortAudio's indices under the held output stream, with no
    exception raised and nothing in the log. Compared on every
    _get_out() so the voice follows the move.

    Exact name wins, then the first case-insensitive substring, so a
    precise name can never be beaten by a loose one. A name matching
    nothing falls back to the default and logs the outputs it saw —
    the voice degrades, it never goes mute.
    """
    global _speaker_device_warned
    want = str(CFG.get("speaker_device", "") or "").strip()
    try:
        devices = sd.query_devices()
    except Exception as e:
        log(f"[mouth] could not list audio devices ({e}) — default output")
        return None
    outs = [(i, d) for i, d in enumerate(devices)
            if d.get("max_output_channels", 0) > 0]
    if want:
        for i, d in outs:
            if d["name"] == want:
                _speaker_device_warned = False
                return i
        low = want.lower()
        for i, d in outs:
            if low in d["name"].lower():
                _speaker_device_warned = False
                return i
        if not _speaker_device_warned:      # once per disappearance
            _speaker_device_warned = True
            log(f"[mouth] speaker_device {want!r} not found — using the "
                f"system default. Outputs I can see: "
                f"{[d['name'] for _, d in outs]}")
    # Default case — resolve to a concrete index so a later renumber is
    # visible as a changed value rather than a constant None.
    try:
        idx = sd.default.device[1]
        if isinstance(idx, (int, np.integer)) and idx >= 0:
            return int(idx)
    except Exception:
        pass
    try:
        name = sd.query_devices(kind="output")["name"]
        for i, d in outs:
            if d["name"] == name:
                return i
    except Exception:
        pass
    return None


class Mouth:
    def __init__(self):
        from backtalk.ducking import Ducker
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        # The one persistent output stream (audio law #1).
        # Worker-thread-only — never touch from other threads.
        self._out: sd.OutputStream | None = None
        self._out_rate: int | None = None
        # The concrete output index the held stream is bound to. A
        # change between sentences means the device list moved under us
        # — a renumbering or a default switch, neither of which raises
        # (see _speaker_index). _get_out rebuilds when it does.
        self._out_dev: int | None = None
        # monotonic time the last reply finished speaking. _get_out uses
        # the gap since then to decide whether to refresh PortAudio's
        # cached device list before the next reply — see the comment
        # there. 0.0 = nothing has played yet.
        self._last_play_end: float = 0.0
        self.ducker = Ducker()  # public: PTT ducks for the USER's voice too
        # Optional () -> bool set by main.py to brain._dirty. When the speech
        # queue drains we normally stamp "idle" -- but a filler line ("on it,
        # let me check") can finish playing while a tool is still running
        # underneath it. If the turn is still live, leave the state alone
        # (brain owns "working"/"thinking" then) instead of flashing idle.
        self.turn_active = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def say(self, text: str):
        """Queue text (split to sentences) for speech."""
        for s in split_sentences(text):
            self._q.put((s, None))

    def say_chunk(self, text: str, directions=None):
        """Queue text as ONE TTS request, no sentence splitting — fuller
        chunks get livelier prosody (single short sentences come out
        dull).

        `directions` are the stage directions this chunk carried. They are
        published on the signal bus when this chunk's audio STARTS, which
        is why they travel with it instead of firing at parse time."""
        text = text.strip()
        if text:
            self._q.put((text, directions or None))

    def shut_up(self):
        """Barge-in: stop current playback and flush everything queued."""
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def shutdown(self):
        """Exit path: stop playback and restore the music SYNCHRONOUSLY
        (the debounced restore timer dies with the process otherwise)."""
        self.shut_up()
        self.ducker.restore_now()

    def wait_done(self, timeout: float | None = None):
        """Block until the queue is drained and playback finished."""
        import time
        deadline = None if timeout is None else time.time() + timeout
        while (not self._q.empty()) or self._speaking.is_set():
            time.sleep(0.05)
            if deadline and time.time() > deadline:
                return

    def _run(self):
        from backtalk import signals
        while True:
            item = self._q.get()
            sentence, directions = item if isinstance(item, tuple) else (item, None)
            if not sentence:
                continue
            self._stop.clear()
            self._speaking.set()
            self.ducker.speech_start()
            signals.static_stop()     # thinking sound dies when speech starts
            signals.set_state("speaking")
            try:
                self._play_stream(sentence, directions)
            except Exception as e:
                log(f"[mouth] synth/play error: {e}")
            finally:
                if self._q.empty():
                    self._speaking.clear()
                    self._last_play_end = time.monotonic()
                    # The reply has genuinely stopped talking, as opposed to
                    # the gap between two sentences of the same reply.
                    signals.reply_done()
                    self.ducker.speech_end()
                    # ...but only settle to idle if the turn is actually done.
                    # A tool still running under a filler line means the brain
                    # owns the state ("working"/"thinking") right now.
                    if not (self.turn_active and self.turn_active()):
                        signals.set_state("idle")

    def _get_out(self, rate: int) -> sd.OutputStream:
        """The long-lived stream (audio law #1). Reopened only when the
        sample rate changes (ElevenLabs 44.1k <-> Kokoro 24k fallback:
        rare, costs at most one blip on the switch), or when the output
        device moves under us — a renumbering or a default switch,
        neither of which raises (see _speaker_index)."""
        # PortAudio snapshots the device list at init and (on Windows
        # especially) never refreshes it on a device add/remove or a
        # default switch, so _speaker_index() below only sees a change
        # after a re-init. On a split mic/speaker rig — the mic on its
        # own USB device, the voice on Bluetooth — switching the output
        # default mid-session breaks nothing the ears would notice, so
        # nothing re-inits and the voice stays on the old device.
        # Bridge that: if this is the first sentence of a reply after a
        # real pause (the user was typing or speaking their next turn),
        # refresh PortAudio first so the voice lands on whatever the OS
        # default is NOW. One faint onset blip, only after an idle gap,
        # landing where a reply is starting anyway. Threshold is
        # config so a machine that swaps devices rarely can raise it.
        idle_s = float(CFG.get("speaker_recheck_idle_s", 10) or 0)
        if (idle_s and self._out is not None and self._last_play_end
                and time.monotonic() - self._last_play_end > idle_s):
            try:
                sd._terminate()
                sd._initialize()
                self._drop_out()   # the held stream died with _terminate;
                #                    forget it so the rebuild below is clean
                log("[mouth] refreshed the audio system after an idle gap "
                    "— the voice will follow a mid-session output switch")
            except Exception as e:
                log(f"[mouth] idle audio refresh failed ({e}) — continuing")
        dev = _speaker_index()
        if (self._out is not None and self._out_rate == rate
                and self._out_dev == dev):
            # Guarded, because the stream can die UNDER us: the ears
            # rebuild the whole audio system to recover from a device
            # change (see ears._reopen_after_device_change), and that
            # closes every open stream including this one. Touching a
            # dead stream raises rather than returning False, so the
            # check has to be the try, not an `if`. Falling through
            # rebuilds it, which is what the rest of this method does.
            try:
                if not self._out.active:
                    self._out.start()
                return self._out
            except Exception:
                log("[mouth] the output stream went away, reopening")
        elif self._out is not None and self._out_dev != dev:
            log(f"[mouth] output device moved ({self._out_dev} -> {dev}) "
                f"— rebuilding the stream")
        self._drop_out()
        self._out = sd.OutputStream(samplerate=rate, channels=1,
                                    dtype="int16", device=dev)
        self._out_rate = rate
        self._out_dev = dev
        self._out.start()
        return self._out

    def _cut(self):
        """Barge-in cut: stop feeding audio and pad the line with a beat
        of silence — the stream itself NEVER stops (an abort+restart here
        re-triggers the onset blip on latch-happy audio setups). Cost:
        the device buffer (~0.1s) plays out after the kill order — half a
        syllable of tail."""
        try:
            zeros = np.zeros(2205, dtype=np.int16)
            for _ in range(3):
                self._out.write(zeros)
        except Exception:
            self._drop_out()

    def _drop_out(self):
        """Close and forget the stream — the next sentence reopens
        fresh. The self-heal path for device errors (interface
        unplugged, audio mixer restarted)."""
        if self._out is not None:
            try:
                self._out.close(ignore_errors=True)
            except Exception:
                pass
        self._out = None
        self._out_rate = None
        self._out_dev = None

    def rebuild_audio(self) -> bool:
        """Tear down and re-initialise PortAudio so the NEXT sentence
        opens its output stream on whatever the OS calls the default
        device right now.

        This is the deliberate, on-request twin of the crash-recovery
        rebuild in ears._reopen_after_device_change. Since _get_out now
        also rebuilds automatically when _speaker_index() reports the
        output device moved, this verb is the manual fallback: it still
        covers the OS default moving to a device that was already
        present (that shifts which index is default, so the same
        comparison fires), and it re-reads PortAudio's own default in
        case _speaker_index couldn't resolve one.

        sd._initialize() re-reads the default device; the held stream is
        left stale on purpose — _get_out() finds it dead on the next
        sentence and rebuilds it fresh, exactly as on a sample-rate
        switch, at the cost of one onset blip.

        Call with the mouth already quiesced (shut_up() then
        wait_done()): terminating PortAudio under a live write is a
        needless race. Returns False (and logs) if the audio system will
        not come back up.
        """
        try:
            sd._terminate()
        except Exception:
            pass                       # already down; re-init is the point
        try:
            sd._initialize()
        except Exception as e:
            log(f"[mouth] could not re-initialise the audio system: {e}")
            return False
        log("[mouth] audio system rebuilt — next sentence opens on the "
            "current default output device")
        return True

    def _play_stream(self, sentence: str, directions=None, block: int = 2205,
                     prebuffer_s: float = 0.75):
        """Stream-synthesize and play with the head-start buffer (audio
        law #2). stop() reacts ~50ms. The sample rate comes from
        whichever engine actually answered."""
        from backtalk import signals
        gen = synth_stream(sentence)
        head: list = []
        banked = 0
        rate = None
        for rate_, pcm in gen:
            rate = rate_
            head.append(pcm)
            banked += len(pcm)
            if banked >= int(rate * prebuffer_s):
                break
        if rate is None:
            return
        try:
            out = self._get_out(rate)
            # AUDIO STARTS HERE: the head buffer is full and the first write
            # is next. Publishing now is what puts a screen cue on the spoken
            # word rather than seconds ahead of it.
            if directions:
                from backtalk import signals as _sig
                _sig.direction(directions)

            def _write(pcm):
                for i in range(0, len(pcm), block):
                    if self._stop.is_set():
                        return False
                    out.write(pcm[i:i + block])
                    # Re-check after the blocking write: a barge-in
                    # landing mid-block must not let feed_waveform
                    # re-assert "speaking" over a fresh "listening".
                    if self._stop.is_set():
                        return False
                    signals.feed_waveform(pcm[i:i + block])
                return True
            for pcm in head:
                if not _write(pcm):
                    self._cut()
                    return
            for _, pcm in gen:
                if not _write(pcm):
                    self._cut()
                    return
        except Exception:
            self._drop_out()
            raise


if __name__ == "__main__":
    m = Mouth()
    m.say(sys.argv[1] if len(sys.argv) > 1 else
          "Voice check. The mouth is alive, and it is very good to be heard.")
    m.wait_done(timeout=60)
