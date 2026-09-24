"""speak_progress_notes: a short thinking block between two tool calls is
spoken; a first-block plan, a long block, or one before text is not.
Scripted fake SDK client from test_brain_fallback.

Run: uv run --with pytest python -m pytest test_progress_notes.py -q
"""
from backtalk.config import CFG
from test_brain_fallback import ResultMessage, StreamEvent, make, turn
from test_brain_fallback import fake  # noqa: F401  (pytest fixture)


def _ev(event):
    e = StreamEvent("")
    e.event = event
    return e


def start(kind):
    return _ev({"type": "content_block_start", "content_block": {"type": kind}})


def think(text):
    return _ev({"type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": text}})


def stop():
    return _ev({"type": "content_block_stop"})


NOTE = "Found the exposure settings. Checking the failed print next."


def _script(first_block_note=False, note=NOTE, then="tool_use"):
    ev = []
    if first_block_note:
        ev += [start("thinking"), think(note), stop()]
    ev += [start("tool_use"), stop()]
    ev += [start("thinking"), think(note), stop()]
    ev += [start(then), stop()]
    ev += [StreamEvent("All done. "), ResultMessage()]
    return [ev]


def _on(monkeypatch, on=True):
    monkeypatch.setitem(CFG, "speak_progress_notes", on)
    monkeypatch.setitem(CFG, "progress_lines", False)


def test_off_by_default_note_not_spoken(fake, monkeypatch):
    _on(monkeypatch, False)
    out = turn(make(fake, _script(), enabled=False))
    assert not any("exposure" in s for s in out)


def test_note_between_tools_is_spoken(fake, monkeypatch):
    _on(monkeypatch)
    out = turn(make(fake, _script(), enabled=False))
    assert any("exposure settings" in s for s in out)


def test_first_block_plan_not_spoken(fake, monkeypatch):
    _on(monkeypatch)
    out = turn(make(fake, _script(first_block_note=True), enabled=False))
    assert sum("exposure settings" in s for s in out) == 1  # only the mid-turn one


def test_long_block_not_spoken(fake, monkeypatch):
    _on(monkeypatch)
    long = "One. Two. Three. Four."
    out = turn(make(fake, _script(note=long), enabled=False))
    assert not any("Three" in s for s in out)


def test_block_before_text_not_spoken(fake, monkeypatch):
    _on(monkeypatch)
    out = turn(make(fake, _script(then="text"), enabled=False))
    assert not any("exposure" in s for s in out)
