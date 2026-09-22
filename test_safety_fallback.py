"""Tests for the safety-classifier auto-switch in brain.WarmBrain: when an
Opus-tier safeguard blocks a turn, switch the session to Sonnet (high
effort), say so, and resend the turn once. Scripted fake SDK client.

Run: uv run --with pytest python -m pytest test_safety_fallback.py -q
"""
import asyncio

from backtalk.config import CFG
from test_brain_fallback import ResultMessage, cloud_ok, cloud_err, make, turn
from test_brain_fallback import fake  # noqa: F401  (pytest fixture)

FLAG_TEXT = ("API Error: Opus 5.5's safeguards flagged this message. "
             "Try rephrasing the request in a new session or change your model.")
SONNET = CFG["sonnet_model"]


class TextBlock:
    def __init__(self, text):
        self.text = text


class AssistantMessage:
    def __init__(self, text, model="<synthetic>"):
        self.content = [TextBlock(text)]
        self.model = model


def flagged_turn():
    return [AssistantMessage(FLAG_TEXT), ResultMessage(is_error=True)]


def refusal_turn():
    r = ResultMessage(is_error=True)
    r.stop_reason = "refusal"
    return [r]


def cmd_ok(text="ok"):
    return [AssistantMessage(text, model="claude"), ResultMessage()]


def test_flag_switches_to_sonnet_and_resends(fake):
    b = make(fake, [flagged_turn(), cmd_ok("Set model to Sonnet"),
                    cmd_ok("Effort set to high"), cloud_ok("Sonnet answer. ")],
             enabled=False)
    out = turn(b, "look at this")
    assert out[0].startswith("Opus's safety filter flagged that")
    assert out[-1] == "Sonnet answer."
    assert b._client.queries == ["look at this", f"/model {SONNET}",
                                 "/effort high", "look at this"]
    assert b._live_model == SONNET


def test_refusal_stop_reason_also_triggers(fake):
    b = make(fake, [refusal_turn(), cmd_ok(), cmd_ok(), cloud_ok("Fine. ")],
             enabled=False)
    out = turn(b)
    assert out[0].startswith("Opus's safety filter flagged that")
    assert out[-1] == "Fine."


def test_flag_again_on_sonnet_stops_without_looping(fake):
    b = make(fake, [flagged_turn(), cmd_ok(), cmd_ok(), flagged_turn()],
             enabled=False)
    out = turn(b)
    assert out[-1].startswith("The safety filter flagged that again")
    assert len(b._client.queries) == 4        # one resend, no more


def test_already_on_sonnet_does_not_switch(fake):
    b = make(fake, [flagged_turn()], enabled=False)
    b._live_model = SONNET
    out = turn(b)
    assert out == ["The safety filter flagged that again. "
                   "Say clear the session to start fresh."]
    assert len(b._client.queries) == 1


def test_failed_switch_says_so(fake):
    b = make(fake, [flagged_turn(), cmd_ok("error: invalid model")],
             enabled=False)
    out = turn(b)
    assert out[-1].startswith("I couldn't switch to Sonnet")
    assert b._live_model != SONNET


def test_model_command_tracks_live_model(fake):
    b = make(fake, [cmd_ok("Set model")], enabled=False)
    asyncio.run(b.command("/model claude-opus-5-5"))
    assert b._live_model == "claude-opus-5-5"


def test_ordinary_error_unchanged(fake):
    b = make(fake, [cloud_err(400, ["invalid request"])], enabled=False)
    out = turn(b)
    assert len(out) == 1 and out[0].startswith("Something went wrong on my end")
