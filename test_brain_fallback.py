"""Tests for the local-fallback wiring in brain.WarmBrain: the SDK client
is a scripted fake (ask_stream dispatches on type(msg).__name__, so the
fakes only need the right class names), the local brain talks to the
scripted fake server from test_local_brain. No SDK, no model needed.

Run: uv run --with pytest python -m pytest test_brain_fallback.py -q
"""
import asyncio

import pytest

import backtalk.brain as brain_mod
from backtalk.brain import WarmBrain
from test_local_brain import Fake, text_reply

# ------------------------------------------------------------- fake SDK bits

class ResultMessage:
    def __init__(self, is_error=False, status=None, errors=None):
        self.is_error = is_error
        self.api_error_status = status
        self.errors = errors or []
        self.usage = {}
        self.session_id = None


class StreamEvent:
    def __init__(self, text):
        self.event = {"type": "content_block_delta",
                      "delta": {"type": "text_delta", "text": text}}


def cloud_ok(text="Cloud answer. "):
    return [StreamEvent(text), ResultMessage()]


def cloud_err(status=529, errors=("overloaded",)):
    return [ResultMessage(is_error=True, status=status, errors=list(errors))]


class FakeClient:
    """Queue of scripted per-turn message lists. An entry of None means
    'never answer' (a dead CLI), which trips the first-message timeout."""
    def __init__(self, turns):
        self.turns = list(turns)
        self.queries = []
        self.interrupts = 0

    async def query(self, text):
        self.queries.append(text)

    async def receive_response(self):
        msgs = self.turns.pop(0) if self.turns else cloud_ok()
        if msgs is None:
            await asyncio.sleep(60)
            return
        for m in msgs:
            yield m

    async def interrupt(self):
        self.interrupts += 1

    async def disconnect(self):
        pass

    async def connect(self):
        pass

    async def get_context_usage(self):
        return None


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def fake():
    f = Fake().start()
    yield f
    f.stop()


def make(fake, client_turns=(), *, enabled=True, n=3, force=False):
    b = WarmBrain(model="test-model", can_use_tool=None, local_fallback={
        "enabled": enabled, "url": fake.url, "model": "fake",
        "retry_cloud_every_n_turns": n, "force": force,
        "stop_when_recovered": True, "max_tool_rounds": 2})
    b._client = FakeClient(client_turns)
    return b


def turn(b, text="hi"):
    async def go():
        return [s async for s in b.ask_stream(text)]
    return asyncio.run(go())


# -------------------------------------------------------------------- tests

def test_disabled_is_inert(fake):
    b = make(fake, [cloud_err()], enabled=False)
    assert b._local is None and b.degraded is False
    out = turn(b)
    assert len(out) == 1 and out[0].startswith("Something went wrong on my end")
    assert fake.requests == []          # the local server never saw a request


def test_outage_error_fails_over_and_announces_once(fake):
    fake.queue = [text_reply("Local one."), text_reply("Local two.")]
    b = make(fake, [cloud_err(529)])
    out = turn(b, "first")
    assert b.degraded is True and b._degraded_turns == 1
    assert out[-1] == "Local one."
    assert any("local backup brain" in s for s in out[:-1])   # announced
    # next turn: straight to local, no cloud query, no second announcement
    out2 = turn(b, "second")
    assert out2 == ["Local two."]
    assert b._client.queries == ["first"]
    assert b._degraded_turns == 2


def test_non_outage_error_does_not_fail_over(fake):
    b = make(fake, [cloud_err(400, ["invalid request"])])
    out = turn(b)
    assert b.degraded is False
    assert out[0].startswith("Something went wrong on my end")
    assert fake.requests == []


def test_outage_by_error_text_without_status(fake):
    fake.queue = [text_reply("Local.")]
    b = make(fake, [cloud_err(None, ["TypeError: fetch failed"])])
    assert turn(b)[-1] == "Local." and b.degraded


def test_probe_succeeds_and_releases_local(fake, monkeypatch):
    fake.queue = [text_reply("L1."), text_reply("L2.")]
    b = make(fake, [cloud_err(503), cloud_ok("Cloud again. ")], n=2)
    stopped, resets = [], []
    monkeypatch.setattr(b._local, "stop_server", lambda: stopped.append(1))
    monkeypatch.setattr(b._local, "reset", lambda: resets.append(1))
    turn(b, "t1")                       # fails over, degraded_turns=1
    assert turn(b, "t2") == ["L2."]     # local, degraded_turns=2
    out = turn(b, "t3")                 # 2 % 2 == 0 -> probe -> cloud answers
    assert out[0] == "Cloud again." and "Claude's back" in out[-1]
    assert b.degraded is False and b._degraded_turns == 0
    assert stopped == [1] and resets == [1]
    assert b._client.queries == ["t1", "t3"]


def test_probe_failure_stays_degraded_without_reannouncing(fake):
    fake.queue = [text_reply("L1."), text_reply("L2.")]
    b = make(fake, [cloud_err(529), cloud_err(529)], n=1)
    turn(b, "t1")
    out = turn(b, "t2")                 # 1 % 1 == 0 -> probe -> still down
    assert out == ["L2."]               # no second "I'm on the local brain"
    assert b.degraded and b._degraded_turns == 2


def test_first_message_timeout_fails_over_quietly(fake, monkeypatch):
    fake.queue = [text_reply("Local after timeout.")]
    monkeypatch.setattr(brain_mod, "_STREAM_FIRST_TIMEOUT", 0.2)
    b = make(fake, [None])              # dead CLI: never answers

    async def fake_start():
        b._client = FakeClient([])
    monkeypatch.setattr(b, "start", fake_start)
    out = turn(b, "q")
    assert out[-1] == "Local after timeout."
    assert not any("sign in again" in s for s in out)   # recover was quiet
    assert b.degraded is True


def test_force_routes_every_turn_local(fake):
    fake.queue = [text_reply("Forced.")]
    b = make(fake, [cloud_ok()], force=True)
    assert turn(b)[-1] == "Forced."
    assert b._client.queries == []


def test_boot_connect_failure_degrades_when_local_up(fake, monkeypatch):
    class DeadSDK:
        def __init__(self, options=None):
            pass

        async def connect(self):
            raise RuntimeError("not signed in")

        async def disconnect(self):
            pass
    monkeypatch.setattr(brain_mod, "ClaudeSDKClient", DeadSDK)
    b = make(fake, [])
    asyncio.run(b.start())
    assert b.degraded is True and b._client is None
    # console commands are refused cleanly rather than crashing on None
    assert asyncio.run(b.command("/effort low")).startswith("error:")


def test_boot_connect_failure_raises_when_local_down(monkeypatch):
    class DeadSDK:
        def __init__(self, options=None):
            pass

        async def connect(self):
            raise RuntimeError("not signed in")

        async def disconnect(self):
            pass
    monkeypatch.setattr(brain_mod, "ClaudeSDKClient", DeadSDK)
    b = WarmBrain(model="m", local_fallback={
        "enabled": True, "url": "http://127.0.0.1:1", "model": "x"})
    with pytest.raises(RuntimeError):
        asyncio.run(b.start())
    assert b.degraded is False


def test_interrupt_after_local_turn_skips_sdk(fake):
    fake.queue = [text_reply("Local.")]
    b = make(fake, [cloud_err(529)])
    turn(b)
    asyncio.run(b.interrupt())
    assert b._client.interrupts == 0
    assert b._dirty is False            # reset_turn has nothing to drain


def test_stop_closes_local(fake, monkeypatch):
    b = make(fake, [])
    stopped = []
    monkeypatch.setattr(b._local, "stop_server", lambda: stopped.append(1))
    asyncio.run(b.stop())
    assert stopped == [1] and b._client is None
