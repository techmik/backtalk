"""Parallel tool calls must queue at the spoken permission gate: one ask
posed at a time, each answered on its own (field case 2026-09-23: five
parallel WebFetches, one card, one "yes" approved only the newest)."""
import asyncio

import pytest

from backtalk import main as main_mod


class _Mouth:
    speaking = False

    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


@pytest.fixture
def gate(monkeypatch):
    for name in ("permission_prompt", "permission_clear", "static_stop",
                 "static_start", "set_state"):
        monkeypatch.setattr(main_mod.signals, name, lambda *a, **k: None)
    monkeypatch.setitem(main_mod._AUTOAPPROVE, "on", False)
    monkeypatch.setitem(main_mod._PERM, "fut", None)
    monkeypatch.setitem(main_mod._PERM, "hinted", True)
    mouth = _Mouth()
    return main_mod.make_permission_gate(mouth), mouth


def _asks(mouth):
    return [s for s in mouth.said if s.startswith("Permission check")]


async def _next_pending():
    for _ in range(500):
        f = main_mod._PERM["fut"]
        if f is not None and not f.done():
            return f
        await asyncio.sleep(0.01)
    raise AssertionError("no ask became pending")


def _calls(g, n):
    return [asyncio.ensure_future(
        g("WebFetch", {"url": f"https://example.com/{i}",
                       "prompt": "summarize"}, None))
        for i in range(n)]


def test_parallel_asks_are_posed_one_at_a_time(gate):
    g, mouth = gate

    async def run():
        tasks = _calls(g, 3)
        answers = ["yes", "no", "yes"]
        for i, ans in enumerate(answers):
            f = await _next_pending()
            # only the asks already posed have been spoken -- never ahead
            assert len(_asks(mouth)) == i + 1
            f.set_result(ans)
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert [r.behavior for r in results] == ["allow", "deny", "allow"]
    assert len(_asks(mouth)) == 3


def test_interrupt_drops_queued_asks_silently(gate):
    g, mouth = gate

    async def run():
        tasks = _calls(g, 3)
        await _next_pending()
        main_mod._deny_pending()
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert [r.behavior for r in results] == ["deny"] * 3
    assert len(_asks(mouth)) == 1          # the queued two never spoke


def test_auto_approve_flipped_mid_queue_releases_the_rest(gate, monkeypatch):
    g, mouth = gate

    async def run():
        tasks = _calls(g, 3)
        f = await _next_pending()
        main_mod._AUTOAPPROVE["on"] = True
        f.set_result("yes")
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert [r.behavior for r in results] == ["allow"] * 3
    assert len(_asks(mouth)) == 1
