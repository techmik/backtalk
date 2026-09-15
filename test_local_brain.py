"""Tests for backtalk.local_brain.LocalBrain against a scripted fake
OpenAI-compatible server (no model needed), plus one live smoke test that
runs only when a real server answers /health on the default URL.

Run: python -m pytest test_local_brain.py -q
"""
import asyncio
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from backtalk.local_brain import LocalBrain

# ---------------------------------------------------------------- fake server

def _sse(events):
    """Encode a list of delta dicts as one SSE body."""
    out = []
    for ev in events:
        out.append("data: " + json.dumps({"choices": [{"index": 0, "delta": ev,
                                                       "finish_reason": None}]}))
    out.append("data: [DONE]")
    return ("\n".join(out) + "\n").encode()


def text_reply(s):
    # split the text so sentence assembly across deltas is exercised
    mid = len(s) // 2
    return _sse([{"role": "assistant"}, {"content": s[:mid]}, {"content": s[mid:]}])


def tool_reply(name, args: dict, call_id="call_1"):
    a = json.dumps(args)
    return _sse([{"role": "assistant"},
                 {"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                  "function": {"name": name, "arguments": a[:3]}}]},
                 {"tool_calls": [{"index": 0,
                                  "function": {"arguments": a[3:]}}]}])


PEG_500 = (500, b'{"error":{"code":500,"message":"The model produced output '
                b'that does not match the expected peg-native format"}}')


class Fake:
    """Queue of scripted responses; records every request body."""
    def __init__(self):
        self.queue = []
        self.requests = []
        self.default = text_reply("Default answer.")

    def start(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 404)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append(body)
                item = fake.queue.pop(0) if fake.queue else fake.default
                if isinstance(item, tuple):
                    code, payload = item
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(item)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.srv.shutdown()


@pytest.fixture
def fake():
    f = Fake().start()
    yield f
    f.stop()


@pytest.fixture
def root():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.realpath(d)


def make(fake, root, gate=None, **cfg):
    c = {"url": fake.url, "model": "fake", "max_tool_rounds": 3, **cfg}
    return LocalBrain(c, name="Jarvis", agent_dir=root, can_use_tool=gate)


def run(brain, utterance):
    async def go():
        out = []
        async for s in brain.ask_stream(utterance):
            out.append(s)
        await brain.close()
        return out
    return asyncio.run(go())


# ---------------------------------------------------------------------- tests

def test_plain_answer_streams_as_sentences(fake, root):
    fake.queue = [text_reply("Hello there. Two sentences here.")]
    assert run(make(fake, root), "hi") == ["Hello there.", "Two sentences here."]
    req = fake.requests[0]
    assert req["chat_template_kwargs"] == {"tools_in_user_message": False}
    assert [t["function"]["name"] for t in req["tools"]] == ["Read", "Write", "Grep", "Glob"]
    assert req["messages"][0]["role"] == "system"
    assert "Jarvis" in req["messages"][0]["content"]


def test_read_tool_round_trip(fake, root):
    with open(os.path.join(root, "note.md"), "w") as f:
        f.write("NEXT UP: rebuild the fallback.")
    fake.queue = [tool_reply("Read", {"file_path": os.path.join(root, "note.md")}),
                  text_reply("Your note says rebuild the fallback.")]
    out = run(make(fake, root), "what does my note say?")
    assert out == ["Your note says rebuild the fallback."]
    # second request carried the assistant tool_calls + the tool result
    msgs = fake.requests[1]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"][0]["function"]["name"] == "Read"
    assert msgs[-1]["role"] == "tool" and "rebuild the fallback" in msgs[-1]["content"]
    assert msgs[-1]["tool_call_id"] == "call_1"


def test_read_outside_root_is_refused(fake, root):
    outside = os.path.join(tempfile.gettempdir(), "definitely_outside.txt")
    fake.queue = [tool_reply("Read", {"file_path": outside}), text_reply("Can't.")]
    run(make(fake, root), "read it")
    assert "outside the folders" in fake.requests[1]["messages"][-1]["content"]


def test_parser_500_retries_without_tools(fake, root):
    fake.queue = [PEG_500, text_reply("Seventeen times three is fifty one.")]
    out = run(make(fake, root), "what is 17 times 3?")
    assert out == ["Seventeen times three is fifty one."]
    assert "tools" in fake.requests[0]
    assert "tools" not in fake.requests[1]


def test_other_error_speaks_one_line(fake, root):
    fake.queue = [(503, b"down")]
    out = run(make(fake, root), "hi")
    assert len(out) == 1 and "isn't answering" in out[0]


def test_tool_round_cap(fake, root):
    open(os.path.join(root, "a.md"), "w").close()
    p = os.path.join(root, "a.md")
    fake.queue = [tool_reply("Read", {"file_path": p}, "c1"),
                  tool_reply("Read", {"file_path": p}, "c2"),
                  tool_reply("Read", {"file_path": p}, "c3"),
                  text_reply("Done looking.")]
    out = run(make(fake, root), "loop")
    assert out == ["Done looking."]
    last = fake.requests[3]
    assert "tools" not in last                      # cut off after 3 rounds
    assert last["messages"][-1]["role"] == "user"   # the "stop using tools" nudge
    assert "Stop using tools" in last["messages"][-1]["content"]


def test_invented_tool_name_returns_error_text(fake, root):
    fake.queue = [tool_reply("calculate", {"a": 1}), text_reply("Fifty one.")]
    run(make(fake, root), "math")
    assert "no such tool" in fake.requests[1]["messages"][-1]["content"]


def test_write_goes_through_gate(fake, root):
    target = os.path.join(root, "new.md")
    asked = []

    class Deny:
        behavior = "deny"

    class Allow:
        behavior = "allow"

    async def gate(tool, tool_input, ctx):
        asked.append((tool, tool_input["file_path"]))
        return Deny() if len(asked) == 1 else Allow()

    fake.queue = [tool_reply("Write", {"file_path": target, "content": "hi"}),
                  text_reply("Denied.")]
    run(make(fake, root, gate=gate), "write it")
    assert asked == [("Write", target)]
    assert not os.path.exists(target)
    assert "Denied" in fake.requests[1]["messages"][-1]["content"]

    fake.queue = [tool_reply("Write", {"file_path": target, "content": "hi"}),
                  text_reply("Written.")]
    run(make(fake, root, gate=gate), "write it again")
    assert open(target).read() == "hi"


def test_grep_and_glob(fake, root):
    os.makedirs(os.path.join(root, "sub"))
    with open(os.path.join(root, "sub", "x.md"), "w") as f:
        f.write("alpha\nBeta line\n")
    with open(os.path.join(root, "y.txt"), "w") as f:
        f.write("beta again\n")
    b = make(fake, root)
    g = b._grep({"pattern": "BETA"})
    assert "x.md:2: Beta line" in g.replace("\\", "/") and "y.txt:1: beta again" in g
    gl = b._globt({"pattern": "**/*.md"})
    assert gl.replace("\\", "/") == "sub/x.md"
    assert "outside" in b._grep({"pattern": "x", "path": tempfile.gettempdir() + "/nope"})


def test_history_trim_keeps_tool_pairs(fake, root):
    b = make(fake, root)
    # 14 messages: the oldest pair is a tool_calls + its result; trimming to
    # 12 would cut the assistant and leave an orphan tool message.
    b._history = [{"role": "user", "content": "q0"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "c", "content": "r"}] + \
                 [{"role": "user", "content": f"q{i}"} if i % 2 else {"role": "assistant", "content": f"a{i}"}
                  for i in range(1, 12)]
    msgs = b._messages()
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] != "tool"
    assert len(msgs) <= 13


def test_fence_rejects_traversal(fake, root):
    b = make(fake, root)
    assert b._fence(os.path.join(root, "..", "escape.txt")) is None
    assert b._fence(os.path.join(root, "ok.txt")) == os.path.join(root, "ok.txt")
    assert b._fence("../escape.txt") is None
    assert b._fence(os.path.join("C:\\" if os.name == "nt" else "/", "etc", "x")) is None
    # "<root>/../x" normalises to a path INSIDE the root -- allowed, not an escape
    assert b._fence(os.path.basename(root) + "/../inside.txt") == os.path.join(root, "inside.txt")


def test_fence_resolves_relative_shapes_a_small_model_produces(fake, root):
    b = make(fake, root)
    want = os.path.join(root, "Active Priorities.md")
    base = os.path.basename(root)
    assert b._fence("Active Priorities.md") == want                  # bare name
    assert b._fence(f"{base}/Active Priorities.md") == want          # root name as prefix
    assert b._fence(f"{base}\\Active Priorities.md") == want         # backslash form
    assert b._fence(f'"{want}"') == want                              # quoted
    # a second root: bare names resolve there only if the file exists
    extra = tempfile.mkdtemp()
    try:
        b2 = LocalBrain({"url": fake.url}, name="J", agent_dir=root, extra_dirs=[extra])
        open(os.path.join(extra, "only-here.md"), "w").close()
        assert b2._fence("only-here.md") == os.path.realpath(os.path.join(extra, "only-here.md"))
        assert b2._fence("new-file.md") == os.path.join(root, "new-file.md")
    finally:
        os.remove(os.path.join(extra, "only-here.md")); os.rmdir(extra)


# ------------------------------------------------------------------ live smoke

def _live_url():
    import httpx
    url = "http://127.0.0.1:8080"
    try:
        return url if httpx.get(f"{url}/health", timeout=2).status_code == 200 else None
    except Exception:
        return None


@pytest.mark.skipif(_live_url() is None, reason="no local llama-server on :8080")
def test_live_server_answers_and_reads(root):
    with open(os.path.join(root, "Active Priorities.md"), "w") as f:
        f.write("# Active Priorities\n- NEXT UP: rebuild the local LLM fallback on llama.cpp.\n")
    b = LocalBrain({"url": _live_url(), "model": "llama-3.1-8b-instruct"},
                   name="Jarvis", agent_dir=root)
    out = run(b, "In one sentence, what is 17 times 3?")
    spoken = " ".join(out).lower().replace("-", " ")
    # the discipline says numbers as words, so accept either form
    assert out and ("51" in spoken or "fifty one" in spoken)
    assert b._history[-1]["role"] == "assistant"
    b2 = LocalBrain({"url": _live_url(), "model": "llama-3.1-8b-instruct"},
                    name="Jarvis", agent_dir=root)
    out2 = run(b2, "Read the file named 'Active Priorities.md' in my main folder and tell me what's next up.")
    assert any(m.get("role") == "tool" for m in b2._history), "expected a Read tool call"
    assert "llama" in " ".join(out2).lower()
