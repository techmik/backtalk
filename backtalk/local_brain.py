"""Break-glass local brain — a small local model behind an OpenAI-compatible
endpoint, used only when the real brain (Claude, via the Agent SDK) is
unreachable. Config block: `local_fallback` in backtalk.json; nothing here
runs unless it is enabled. See brain.WarmBrain for the hand-off.

Same contract as WarmBrain.ask_stream: an async generator that yields
complete spoken sentences, siphons fenced code to the transcript bus, and
sets `_turn_had_code` for main.speak_reply. Same transcript/state signals
as the SDK path, so a face or a chatbox shows the activity identically.

Deliberately NOT a second assistant: no memory files, no skills, no MCP,
no subagents, no web. Four file tools fenced to agent_dir + extra_dirs,
writes routed through the same spoken permission gate the SDK path uses.

Tested against llama.cpp `llama-server` + Llama 3.1 8B Instruct
(2026-09-15). Three things that server/model combination needs, all
handled here and worth knowing before touching the request shape:

  1. Send chat_template_kwargs={"tools_in_user_message": false}. Meta's
     Llama 3.1 template otherwise injects "please respond with a JSON for
     a function call" into EVERY user turn -> a tool call on every
     question, and a re-call after the tool result.
  2. The template stamps "Environment: ipython" whenever tools are
     present, so the model invents a `calculate` tool for arithmetic. The
     system prompt names the only tools that exist and says there is no
     code interpreter.
  3. An invented tool name is an HTTP 500 from llama.cpp ("does not match
     the expected peg-native format", upstream issue 26381), not a
     graceful miss. That 500 is caught and the turn retried once with
     tools withheld.
"""
import asyncio
import glob as _glob
import json
import os
import shlex
import subprocess
import time
from datetime import datetime

import httpx

from backtalk import signals
from backtalk.config import CFG, REPO
from backtalk.vlog import log

# Reused from the SDK brain so both paths split speech from code the same
# way and describe tool calls on the bus with the same words.
from backtalk.brain import _StreamSplitter, _drain_think, _squish, _tool_summary

# Tool output caps: everything a tool returns goes straight back into an
# 8K-token context. Mirrors brain._BASH_OUTPUT_CHAR_LIMIT in spirit.
_READ_CHAR_LIMIT = 4000
_RESULT_CHAR_LIMIT = 1500
# History kept per request: the system prompt plus this many messages,
# and never more than this many characters of them.
_HISTORY_MSGS = 12
_HISTORY_CHARS = 18000
# Gemma 4 thinks before it answers and hidden reasoning tokens count against
# this cap: at 500 a multi-digit multiplication came back EMPTY (reasoning
# ate the whole budget, finish_reason=length); it needed ~1000. The context
# is only 8K and can't grow (VRAM), so this leaves ~6K for the prompt.
_MAX_TOKENS = 2048
_PEG_500 = "peg-native format"


# The tools are named and shaped like Claude Code's own (Read/Write/Grep/
# Glob with file_path/content/pattern/path) ON PURPOSE: brain._tool_summary
# and main's permission gate already know how to describe them aloud, so
# the transcript and the spoken "I want to write the file X" line come
# out identical to the SDK path with zero new phrasing code.
TOOLS = [
    {"type": "function", "function": {
        "name": "Read",
        "description": "Read a text file. Returns its contents (long files are cut).",
        "parameters": {"type": "object",
                       "properties": {"file_path": {"type": "string"}},
                       "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "Write",
        "description": "Create or overwrite a text file with the given content.",
        "parameters": {"type": "object",
                       "properties": {"file_path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["file_path", "content"]}}},
    {"type": "function", "function": {
        "name": "Grep",
        "description": "Search text files under a folder for lines containing a phrase (case-insensitive).",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                      "path": {"type": "string",
                                               "description": "Folder to search; defaults to the main folder."}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "Glob",
        "description": "List files matching a glob pattern such as **/*.md, relative to a folder.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                      "path": {"type": "string",
                                               "description": "Folder to search; defaults to the main folder."}},
                       "required": ["pattern"]}}},
]
_TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".obsidian"}
_TEXT_EXT = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml",
             ".ini", ".cfg", ".sh", ".ps1", ".csv", ".html", ".css", ".js"}


class _Ctx:
    """Minimal stand-in for the SDK's tool-permission context: main's
    gate only reads display_name / description off it."""
    def __init__(self, name, desc):
        self.display_name = name
        self.description = desc


def _compact_discipline(name: str, roots: list[str]) -> str:
    """The spoken-delivery rules, cut to what an 8B model can hold
    alongside a conversation in 8K tokens. The full DISCIPLINE plus a
    CLAUDE.md would not fit, so the character is one line and the
    medium is the essentials."""
    # Full paths, not basenames: the model echoes what it was shown, and a
    # basename alone came back as a relative prefix that had to be guessed
    # at. The prompt is never spoken, so the path rule doesn't apply here.
    folders = "; ".join(roots)
    # Wording matters here more than usual: an early draft said "you are
    # running on a small local model ... if asked, say so plainly" and
    # the 8B model leaned on it as an excuse -- "capital of France?" got
    # "I'm on a small local model so I don't have much information"
    # (live, 2026-09-15). Now: answer first, mention the backup only if
    # asked about it.
    return (
        f"You are {name}, a voice assistant. Answer questions directly "
        "and confidently from what you know. You happen to be the backup "
        "brain today, but that's only worth mentioning if someone asks "
        "what you are or why you seem different; never use it as a "
        "reason not to answer. Your reply is spoken aloud by a "
        "text-to-speech engine: write like you talk, contractions and "
        "short sentences, a few sentences at most. No markdown, no "
        "lists, no emoji, no URLs. If you must show a file's contents, "
        "code, or a command, put it inside a triple-backtick fence on "
        "its own lines and say one short sentence about it; the fence "
        "is shown on screen, never spoken. Never say a file path out "
        "loud, name the file instead. Say numbers the way people say "
        "them. "
        f"Tools: the ONLY functions that exist are {', '.join(_TOOL_NAMES)}, "
        f"and they only reach files inside these folders: {folders}. Call "
        "one only when the answer needs a file you have not seen; "
        "answer everything else directly. Never invent or call any other "
        "function name. You have no calculator and no code interpreter; "
        "do arithmetic yourself in plain text. "
        f"Today is {datetime.now():%A %d %B %Y}."
    )


class LocalBrain:
    def __init__(self, cfg: dict, *, name: str, agent_dir: str,
                 extra_dirs=(), can_use_tool=None):
        self.url = str(cfg.get("url") or "http://127.0.0.1:8080").rstrip("/")
        self.model = str(cfg.get("model") or "local")
        self.start_cmd = str(cfg.get("start_cmd") or "")
        self.start_timeout = float(cfg.get("start_timeout_s") or 45)
        self.max_tool_rounds = int(cfg.get("max_tool_rounds") or 6)
        self.name = name
        self._gate = can_use_tool
        self._roots = [os.path.realpath(agent_dir)] + \
                      [os.path.realpath(d) for d in (extra_dirs or ())]
        self._system = _compact_discipline(name, self._roots)
        self._history: list[dict] = []
        self._proc: subprocess.Popen | None = None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5))
        self._turn_had_code = False
        self.turns = 0

    # ---- server lifecycle -------------------------------------------------

    async def healthy(self) -> bool:
        try:
            r = await self._http.get(f"{self.url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    async def ensure_up(self) -> bool:
        """True if the server answers, starting it via start_cmd when it
        doesn't and a command is configured. Bounded by start_timeout."""
        if await self.healthy():
            return True
        if not self.start_cmd:
            log("[local] server not answering and no start_cmd configured")
            return False
        if self._proc is None or self._proc.poll() is not None:
            logf = open(os.path.join(REPO, "logs", "local_server.log"),
                        "ab", buffering=0)
            kw = {}
            if os.name == "nt":
                # NOT DETACHED_PROCESS: a `powershell -File` launcher with
                # no console exits at once with code 0 and never starts
                # the server (measured 2026-09-15). No-window + own
                # process group runs it invisibly and lets stop_server's
                # taskkill /T take the whole tree down.
                kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                       | subprocess.CREATE_NO_WINDOW)
            else:
                kw["start_new_session"] = True
            log(f"[local] starting server: {self.start_cmd[:120]}")
            # No shell: on Windows CreateProcess parses the string itself,
            # elsewhere split it into argv. start_cmd is the operator's
            # own config line, but a shell buys nothing here.
            argv = self.start_cmd if os.name == "nt" else shlex.split(self.start_cmd)
            self._proc = subprocess.Popen(argv, stdout=logf, stderr=logf, **kw)
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            if await self.healthy():
                log("[local] server up")
                return True
            if self._proc.poll() is not None:
                log(f"[local] server process exited early "
                    f"(code {self._proc.returncode})")
                return False
        log(f"[local] server not up after {self.start_timeout:.0f}s")
        return False

    def stop_server(self):
        """Stop a server WE started (never one that was already running).
        start_cmd is usually a launcher script wrapping the real server
        process, so kill the whole tree, not just the parent."""
        p, self._proc = self._proc, None
        if p is None or p.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True, timeout=10)
            else:
                import signal
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            log("[local] server stopped")
        except Exception as e:
            log(f"[local] stop failed: {e!r}")

    async def close(self):
        await self._http.aclose()

    def reset(self):
        self._history.clear()

    # ---- the turn ---------------------------------------------------------

    def _messages(self) -> list[dict]:
        """System prompt + a trimmed tail of the history. Trims whole
        messages from the front, then drops any orphaned tool results so
        a tool_calls message is never separated from its results (the
        Llama template needs the pair)."""
        hist = self._history[-_HISTORY_MSGS:]
        while len(hist) > 1 and sum(len(json.dumps(m)) for m in hist) > _HISTORY_CHARS:
            hist = hist[1:]
        while hist and hist[0].get("role") == "tool":
            hist = hist[1:]
        return [{"role": "system", "content": self._system}] + hist

    async def _request(self, with_tools: bool):
        """One streamed chat completion. Returns (text_chunks_iter) as an
        async generator of ("text", str) / ("thinking", str) /
        ("finish", str) / ("tool_calls", list) / ("error", str) events.
        "thinking" is the model's reasoning (llama-server streams it as
        delta.reasoning_content, separate from the spoken text); "finish"
        is the stream's finish_reason. Tool-call argument fragments are
        stitched by index the way the OpenAI stream format delivers them."""
        body = {
            "model": self.model,
            "messages": self._messages(),
            "stream": True,
            "temperature": 0.3,
            "max_tokens": _MAX_TOKENS,
            "chat_template_kwargs": {"tools_in_user_message": False},
        }
        if with_tools:
            body["tools"] = TOOLS
        calls: dict[int, dict] = {}
        async with self._http.stream("POST", f"{self.url}/v1/chat/completions",
                                     json=body) as r:
            if r.status_code != 200:
                raw = (await r.aread()).decode(errors="replace")
                yield ("error", f"HTTP {r.status_code}: {raw[:300]}")
                return
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    yield ("thinking", delta["reasoning_content"])
                if delta.get("content"):
                    yield ("text", delta["content"])
                if choice.get("finish_reason"):
                    yield ("finish", choice["finish_reason"])
                for tc in delta.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    slot = calls.setdefault(i, {"id": tc.get("id") or f"call_{i}",
                                                "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
        if calls:
            yield ("tool_calls", [calls[i] for i in sorted(calls)])

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model."""
        self._turn_had_code = False
        self.turns += 1

        def _emit_code(block):
            log(f"[code] block ({len(block)} chars) -> screen only")
            signals.transcript("code", block)
            self._turn_had_code = True

        split = _StreamSplitter(_emit_code)
        self._history.append({"role": "user", "content": utterance})
        with_tools = True
        rounds = 0
        while True:
            text = ""
            tool_calls = None
            error = None
            finish = None
            think_buf = ""
            try:
                async for kind, val in self._request(with_tools):
                    if kind == "text":
                        text += val
                        for s in split.feed(val):
                            yield s
                    elif kind == "thinking":
                        # Screen-only, like the SDK path's thinking blocks:
                        # rides the transcript bus as its own "thinking" role
                        # and is never yielded, so the mouth never speaks it.
                        think_buf += val
                        segs, think_buf = _drain_think(think_buf)
                        for seg in segs:
                            log(f"[think] {seg}")
                            signals.transcript("thinking", seg)
                    elif kind == "finish":
                        finish = val
                    elif kind == "tool_calls":
                        tool_calls = val
                    else:
                        error = val
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)[:160]}"
            if think_buf.strip():
                seg = think_buf.strip()
                log(f"[think] {seg}")
                signals.transcript("thinking", seg)
            if error:
                if _PEG_500 in error and with_tools:
                    # The model invented a function name and llama.cpp
                    # refused to parse it. Ask again with no tools on
                    # offer: it answers in prose instead.
                    log("[local] parser 500 (invented tool) -> retry without tools")
                    with_tools = False
                    continue
                log(f"[local] request failed: {error}")
                for s in split.close():
                    yield s
                yield ("The local brain isn't answering either. "
                       "Try me again in a moment.")
                return
            if not tool_calls and not text.strip() and finish == "length":
                # A thinking model spent the whole token budget on reasoning
                # and never got to the answer (seen live on multi-digit
                # arithmetic). Say so instead of going silent.
                log("[local] empty reply: reasoning used the whole token budget")
                for s in split.close():
                    yield s
                line = ("I got lost thinking that one through. "
                        "Try asking it another way.")
                self._history.append({"role": "assistant", "content": line})
                yield line
                return
            if not tool_calls:
                break
            # -- tool round --
            for s in split.flush():
                yield s
            self._history.append({"role": "assistant", "content": text,
                                  "tool_calls": [
                                      {"id": c["id"], "type": "function",
                                       "function": {"name": c["name"],
                                                    "arguments": c["arguments"]}}
                                      for c in tool_calls]})
            rounds += 1
            for c in tool_calls:
                result = await self._run_tool(c["name"], c["arguments"])
                self._history.append({"role": "tool", "tool_call_id": c["id"],
                                      "content": result})
            if rounds >= self.max_tool_rounds:
                log(f"[local] tool round cap ({self.max_tool_rounds}) hit")
                self._history.append({"role": "user", "content":
                                      "Stop using tools now and answer with "
                                      "what you have."})
                with_tools = False
            signals.set_state("thinking")
        self._history.append({"role": "assistant", "content": text})
        for s in split.close():
            yield s

    # ---- tools ------------------------------------------------------------

    def _inside(self, p: str) -> str | None:
        p = os.path.realpath(p)
        for root in self._roots:
            if p == root or p.startswith(root + os.sep):
                return p
        return None

    def _fence(self, path: str) -> str | None:
        """Real path if it sits inside an allowed root, else None.

        An 8B model hands back paths in whatever shape it last saw: the
        absolute path, a bare filename, or the root folder's own name as
        a prefix ("MyJarvisVault/Active Priorities.md" -- seen live
        2026-09-15). Resolve leniently: absolute as-is; relative against
        each root; and a leading component that names a root is taken
        as that root. Escapes ("..", other drives) still fail the check."""
        path = str(path or "").strip().strip('"').strip("'")
        if not path:
            return None
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return self._inside(path)
        norm = path.replace("\\", "/")
        head, _, rest = norm.partition("/")
        for root in self._roots:
            if rest and head == os.path.basename(root):
                hit = self._inside(os.path.join(root, rest))
                if hit:
                    return hit
        # Prefer wherever the file already exists; a new file lands in
        # the primary root.
        for root in self._roots:
            hit = self._inside(os.path.join(root, path))
            if hit and os.path.exists(hit):
                return hit
        return self._inside(os.path.join(self._roots[0], path))

    def _default_root(self, path: str | None) -> str | None:
        return self._fence(path) if path else self._roots[0]

    async def _run_tool(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if raw_args else {}
            if not isinstance(args, dict):
                args = {}
        except ValueError:
            args = {}
        signals.transcript("tool", _tool_summary(name, args))
        signals.set_state("working")
        log(f"[tool] {_tool_summary(name, args)}")
        try:
            if name == "Read":
                out = self._read(args)
            elif name == "Write":
                out = await self._write(args)
            elif name == "Grep":
                out = self._grep(args)
            elif name == "Glob":
                out = self._globt(args)
            else:
                out = f"Error: no such tool {name!r}. Only {', '.join(_TOOL_NAMES)} exist."
        except Exception as e:
            out = f"Error: {type(e).__name__}: {str(e)[:200]}"
        signals.transcript("tool-result", _squish(out, 200))
        return out

    @staticmethod
    def _nearest_existing(p: str) -> tuple[str | None, list[str]]:
        """A small model guesses names: wrong case, wrong extension
        ("Active Priorities.txt" for a .md, seen live 2026-09-15). If the
        exact path is missing, look in the same folder for a name that
        matches case-insensitively, then for the same stem with any
        extension. One match -> use it. Several -> hand them back so the
        model can pick. Never leaves the folder."""
        if os.path.isfile(p):
            return p, []
        d, want = os.path.dirname(p), os.path.basename(p)
        if not os.path.isdir(d):
            return None, []
        names = [n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n))]
        ci = [n for n in names if n.lower() == want.lower()]
        if len(ci) == 1:
            return os.path.join(d, ci[0]), []
        stem = os.path.splitext(want)[0].lower()
        same_stem = [n for n in names if os.path.splitext(n)[0].lower() == stem]
        if len(same_stem) == 1:
            return os.path.join(d, same_stem[0]), []
        return None, sorted(same_stem or ci)[:8]

    def _read(self, a) -> str:
        p = self._fence(str(a.get("file_path", "")))
        if not p:
            return "Error: that file is outside the folders you may read."
        hit, near = self._nearest_existing(p)
        if not hit:
            if near:
                return ("Error: no such file. Did you mean one of: "
                        + ", ".join(near) + "?")
            return ("Error: no such file. Use Glob with a pattern like "
                    "**/*name* to find the exact filename.")
        p = hit
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read(_READ_CHAR_LIMIT + 1)
        if len(text) > _READ_CHAR_LIMIT:
            text = text[:_READ_CHAR_LIMIT] + "\n...[cut: file continues]"
        return text or "(empty file)"

    async def _write(self, a) -> str:
        path = str(a.get("file_path", ""))
        content = str(a.get("content", ""))
        p = self._fence(path)
        if not p:
            return "Error: that file is outside the folders you may write."
        if self._gate is not None:
            res = await self._gate("Write", {"file_path": p, "content": content},
                                   _Ctx("Write", "write a file"))
            if getattr(res, "behavior", "deny") != "allow":
                return "Denied: the user did not approve writing that file."
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {os.path.basename(p)}."

    def _grep(self, a) -> str:
        pat = str(a.get("pattern", "")).lower()
        root = self._default_root(a.get("path"))
        if not pat:
            return "Error: empty pattern."
        if not root:
            return "Error: that folder is outside the folders you may search."
        hits, n = [], 0
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in _TEXT_EXT:
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if pat in line.lower():
                                rel = os.path.relpath(fp, root)
                                hits.append(f"{rel}:{i}: {_squish(line, 120)}")
                                n += 1
                                if n >= 30:
                                    break
                except OSError:
                    continue
                if n >= 30:
                    break
            if n >= 30:
                break
        out = "\n".join(hits) or "No matches."
        return out[:_RESULT_CHAR_LIMIT]

    def _globt(self, a) -> str:
        pat = str(a.get("pattern", ""))
        root = self._default_root(a.get("path"))
        if not pat:
            return "Error: empty pattern."
        if not root:
            return "Error: that folder is outside the folders you may list."
        found = _glob.glob(os.path.join(root, pat), recursive=True)
        found = [os.path.relpath(f, root) for f in found
                 if not any(part in _SKIP_DIRS for part in f.split(os.sep))]
        out = "\n".join(sorted(found)[:50]) or "No matches."
        return out[:_RESULT_CHAR_LIMIT]
