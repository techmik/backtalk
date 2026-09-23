"""Tests for spoken progress lines: brain._tool_spoken's phrase mapping and
the gate in ask_stream (config switch, gap since last speech, no repeats).
Scripted fake SDK client from test_brain_fallback.

Run: uv run --with pytest python -m pytest test_progress_lines.py -q
"""
from backtalk import brain as brain_mod
from backtalk.brain import Progress, _tool_spoken
from backtalk.config import CFG
from test_brain_fallback import ResultMessage, StreamEvent, make, turn
from test_brain_fallback import fake  # noqa: F401  (pytest fixture)


class ToolUseBlock:
    def __init__(self, name, inp):
        self.name = name
        self.input = inp


class AssistantMessage:
    def __init__(self, *blocks):
        self.content = list(blocks)


def block_stop():
    ev = StreamEvent("")
    ev.event = {"type": "content_block_stop"}
    return ev


def tool(name, **inp):
    return AssistantMessage(ToolUseBlock(name, inp))


# ------------------------------------------------------------- phrase mapping

def test_read_names_the_file():
    assert _tool_spoken("Read", {"file_path": r"C:\v\Active Priorities.md"}) \
        == "Reading Active Priorities."


def test_date_file_is_the_daily_note():
    assert _tool_spoken("Read", {"file_path": "/n/2026-09-23.md"}) \
        == "Reading the daily note."


def test_edit_and_write_say_updating():
    assert _tool_spoken("Edit", {"file_path": "/x/brain.py"}) == "Updating brain."
    assert _tool_spoken("Write", {"file_path": "/x/test_progress_lines.py"}) \
        == "Updating test progress lines."


def test_search_vault_vs_files():
    assert _tool_spoken("Grep", {"path": r"C:\Users\M\MyJarvisVault"}) \
        == "Searching the vault."
    assert _tool_spoken("Glob", {"pattern": "*.py"}) == "Searching the files."
    assert _tool_spoken("mcp__plugin_qmd_qmd__query", {}) == "Searching the vault."


def test_bash_uses_clean_description_only():
    assert _tool_spoken("Bash", {"description": "Run backtalk tests"}) \
        == "Run backtalk tests."
    assert _tool_spoken("Bash", {"description": "Read C:/x/y"}) == "Running a command."
    assert _tool_spoken("Bash", {}) == "Running a command."


def test_web_skill_agent():
    assert _tool_spoken("WebSearch", {"query": "q"}) == "Searching the web."
    assert _tool_spoken("WebFetch", {"url": "u"}) == "Pulling up a page."
    assert _tool_spoken("Skill", {"skill": "session-report:session-report"}) \
        == "Running the session report skill."
    assert _tool_spoken("Agent", {}) == "Handing that to a helper."


def test_plumbing_tools_stay_silent():
    assert _tool_spoken("ToolSearch", {"query": "x"}) is None
    assert _tool_spoken("TodoWrite", {}) is None
    assert _tool_spoken("mcp__other__thing", {}) is None


# ---------------------------------------------------------------------- gate

def _on(monkeypatch, gap=6):
    monkeypatch.setitem(CFG, "progress_lines", True)
    monkeypatch.setitem(CFG, "progress_gap_s", gap)


def test_off_by_default_yields_nothing_extra(fake, monkeypatch):
    monkeypatch.setitem(CFG, "progress_lines", False)
    b = make(fake, [[tool("Read", file_path="/a.md"),
                     StreamEvent("Done. "), ResultMessage()]], enabled=False)
    out = turn(b)
    assert out == ["Done."] and not any(isinstance(s, Progress) for s in out)


def test_first_tool_of_silent_turn_speaks(fake, monkeypatch):
    _on(monkeypatch)
    b = make(fake, [[tool("Read", file_path="/a/Notes.md"),
                     StreamEvent("Done. "), ResultMessage()]], enabled=False)
    out = turn(b)
    assert isinstance(out[0], Progress) and out[0] == "Reading Notes."
    assert out[-1] == "Done."


def test_gap_suppresses_back_to_back_lines(fake, monkeypatch):
    _on(monkeypatch, gap=6)
    b = make(fake, [[tool("Read", file_path="/a/One.md"),
                     tool("Grep", pattern="x"),
                     StreamEvent("Done. "), ResultMessage()]], enabled=False)
    out = turn(b)
    assert [s for s in out if isinstance(s, Progress)] == ["Reading One."]


def test_recent_sentence_suppresses_line(fake, monkeypatch):
    _on(monkeypatch, gap=6)
    b = make(fake, [[StreamEvent("Let me check. "), block_stop(),
                     tool("Read", file_path="/a/One.md"),
                     StreamEvent("Done. "), ResultMessage()]], enabled=False)
    out = turn(b)
    assert out[0] == "Let me check."
    assert not any(isinstance(s, Progress) for s in out)


def test_same_phrase_never_repeats(fake, monkeypatch):
    _on(monkeypatch, gap=0)
    b = make(fake, [[tool("Grep", pattern="a"), tool("Glob", pattern="b"),
                     tool("WebSearch", query="c"),
                     StreamEvent("Done. "), ResultMessage()]], enabled=False)
    out = turn(b)
    assert [s for s in out if isinstance(s, Progress)] == \
        ["Searching the files.", "Searching the web."]


def test_progress_does_not_count_as_a_reply(fake, monkeypatch):
    # a turn that only ran a tool and said nothing still gets the
    # "I got nothing back" line -- progress isn't an answer
    _on(monkeypatch)
    b = make(fake, [[tool("Read", file_path="/a/One.md"), ResultMessage()]],
             enabled=False)
    out = turn(b)
    assert out[0] == "Reading One."
    assert any("got nothing back" in s for s in out[1:])


def test_brain_module_exports_progress():
    assert issubclass(brain_mod.Progress, str)
