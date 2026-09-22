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
"""The warm brain — a persistent Claude session via the Agent SDK,
streaming.

One ClaudeSDKClient lives for the whole voice session: no per-turn
process spawn, no per-turn context reload. Partial-message streaming
means sentences are yielded the moment they're complete, so the mouth
starts speaking while the rest of the thought is still forming.

The session's cwd is YOUR agent's folder (agent_dir in backtalk.json) —
whatever CLAUDE.md lives there defines who is speaking. backtalk adds
only the spoken-delivery discipline (config.DISCIPLINE): the medium,
never the character.
"""
import asyncio
import os
import re
import warnings
from datetime import datetime

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

try:
    from claude_agent_sdk import CanUseToolShadowedWarning
except ImportError:                       # older SDKs: nothing to silence
    CanUseToolShadowedWarning = None

from backtalk import signals
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log
from backtalk import signals

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

# A verbose Bash result (git output, a build log, a long script's stdout)
# feeds straight back into the model's next turn as prompt tokens -- in a
# voice session that's extra prompt-processing time before the reply can
# even start, i.e. more audible dead air on top of whatever the thinking
# sound already covers. Trim before it goes back, not after.
_BASH_OUTPUT_CHAR_LIMIT = 1500  # keep first ~20-30 lines, well under a
                                # typical short-command's output
_bash_shape_logged = False     # log the real tool_response shape once,
                                # on the first real Bash call -- the shape
                                # was unverified as of 2026-09-14 (see the
                                # Backtalk.md vault note)


async def _trim_bash_output(input_data, tool_use_id, context):
    """PostToolUse hook: cap Bash tool output before it re-enters the
    model's context. Shape of tool_response isn't documented per-tool by
    the SDK (typed as Any) -- handle a plain string or a dict with a
    stdout/output/content key defensively, and never raise: a truncation
    feature must not be the thing that breaks a tool call."""
    global _bash_shape_logged
    if input_data.get("tool_name") != "Bash":
        return {}
    try:
        resp = input_data.get("tool_response")
        if not _bash_shape_logged:
            _bash_shape_logged = True
            log(f"[hook] first real Bash tool_response shape: "
                f"type={type(resp).__name__} "
                f"repr={str(resp)[:300]!r}")
        if isinstance(resp, str):
            text, container = resp, None
        elif isinstance(resp, dict):
            key = next((k for k in ("stdout", "output", "content")
                        if isinstance(resp.get(k), str)), None)
            if not key:
                return {}
            text, container = resp[key], key
        else:
            return {}
        if len(text) <= _BASH_OUTPUT_CHAR_LIMIT:
            return {}
        trimmed = (text[:_BASH_OUTPUT_CHAR_LIMIT]
                   + f"\n...[{len(text) - _BASH_OUTPUT_CHAR_LIMIT} more "
                     "chars truncated for voice session]")
        updated = trimmed if container is None else {**resp, container: trimmed}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": updated,
            }
        }
    except Exception as e:
        log(f"[hook] _trim_bash_output failed, passing through: {e}")
        return {}

# ask_stream never bounded the FIRST message out of receive_response the
# way command() and reset_turn() do, so a turn that produced nothing at
# all (a hung CLI subprocess on an expired sign-in — field case
# 2026-09-09) sat silent forever with the face idle. Only the first
# message is bounded: once messages are flowing a long gap is a real
# tool call (an MCP 3D generation, a big Bash run, a pending permission
# ask), not a stall. A first-message timeout raises into the same
# recovery path a transport death takes.
_STREAM_FIRST_TIMEOUT = 30

# ---- local fallback (config: local_fallback, off by default) ------------
# While degraded, a periodic cloud probe tries Claude first; a dead cloud
# is bounded tighter than a normal turn so the probe costs the person
# seconds, not half a minute, before the local brain answers instead.
_PROBE_FIRST_TIMEOUT = 12
# A connect that hangs is treated as an outage only when a local brain can
# take over; without one the old unbounded connect (and main's boot guard)
# apply exactly as before.
_CONNECT_TIMEOUT_DEGRADABLE = 60
# What counts as "Claude is unreachable" on an error ResultMessage: the
# API refusing/overloaded/limited, auth gone, or the CLI unable to reach
# the network. A 400-class content error is NOT an outage -- the local
# brain must not take a turn Claude merely rejected.
_OUTAGE_STATUSES = {401, 403, 408, 429, 500, 502, 503, 504, 529}
_OUTAGE_HINTS = ("overloaded", "rate limit", "rate_limit", "usage limit",
                 "out of usage", "fetch failed", "econn", "enotfound",
                 "etimedout", "network", "connection", "unable to reach",
                 "not signed in", "not logged in", "authentication",
                 "unauthorized")


# Opus-tier safety classifiers can block a normal turn (a false positive —
# seen 2026-09-22, category reasoning_extraction, right after reading a
# chatbox screenshot). The CLI reports it as a synthetic assistant text
# containing this phrase; the block persists for every later turn in the
# same session. Recovery per the error itself: change the model.
_SAFETY_FLAG = "safeguards flagged"


def _is_outage(status, errs: str) -> bool:
    if status in _OUTAGE_STATUSES:
        return True
    e = (errs or "").lower()
    return any(h in e for h in _OUTAGE_HINTS)

# Titles / abbreviations that carry a "." mid-sentence. A sentence break
# found right after one of these ("Dr. Johnson", "at 9 a.m. Tuesday",
# "e.g. this one") is a false split -- the mouth would ship "Dr." as its
# own clipped chunk. Compared case-insensitively against the dotted token
# just before the break; lone initials ("J." in "J. R. R. Tolkien") are
# suppressed the same way. Only the "." case is guarded -- "!" and "?"
# after an abbreviation effectively never happen.
_ABBREV = frozenset("""
    dr mr mrs ms mx prof rev fr sr jr hon gen sen rep gov
    capt sgt lt col st mt
    vs etc et al vol fig pp ca approx dept est inc corp co ltd
    e.g i.e cf ibid
    a.m p.m u.s u.k u.n e.u d.c ph.d b.a m.a m.d
""".split())


def _is_abbrev_dot(head: str) -> bool:
    """True if `head` (text up to and including a ".") ends on an
    abbreviation or a lone initial rather than a real sentence."""
    parts = head.split()
    if not parts:
        return False
    bare = parts[-1].lower().rstrip(".")
    if bare in _ABBREV:
        return True
    return len(bare) == 1 and bare.isalpha()      # lone initial: "J."


def _first_real_break(prose: str):
    """Index just past the first *true* sentence end in `prose`, or None.
    Skips a break that lands right after an abbreviation or a lone
    initial so "Dr. Johnson" is never spoken as two chunks."""
    for m in _SENTENCE_END.finditer(prose):
        head = prose[:m.start()]              # up to & including . ! or ?
        if head.endswith(".") and _is_abbrev_dot(head):
            continue
        return m.end()
    return None


# Thinking flushes to the transcript bus on a newline OR a sentence end,
# so a long reasoning pause streams to a dashboard as it forms instead of
# landing in one lump at block end (glued to the first spoken sentence).
_THINK_FLUSH = re.compile(r"\n|(?<=[.!?])[ \t]")

# A "sentence" the mouth must never speak because it is really a pasted
# block, not speech. Two triggers: a filesystem-path-like token (two or
# more segments joined by "/" or "\"), or a length no spoken sentence
# reaches. One bare filename ("check ears.py") stays speakable; a path,
# a URL, or a wall of them does not. This is the mechanical backstop for
# when the model ignores DISCIPLINE's "never speak a file path" and
# narrates a file dump -- such lines are rerouted to the screen-only
# code role by _StreamSplitter._route_prose.
_PATHLIKE = re.compile(r"[^\s/\\]+[/\\][^\s/\\]+[/\\][^\s]*")
# A "path" made only of digits and separators is a date ("9/22/2026") or a
# ratio ("24/7/365"), which the model is right to speak. Found 2026-09-04:
# the daily spoken recap's calendar line was being diverted to the screen.
_NUMERIC_PATH = re.compile(r"[\d.,/\\]+")
# A slash BETWEEN two backticked words ("`model`/`deep_model`") is a list
# of alternatives, not a path. Found 2026-09-22: such a list hid the very
# question the agent was asking. A real path keeps its slashes inside one
# backtick pair ("`backtalk/backtalk/main.py`"), so it still reads as a path.
_TICK_LIST_SEP = re.compile(r"`\s*/\s*`")
_UNSPEAKABLE_LEN = 400


def _looks_unspeakable(s: str) -> bool:
    if len(s) > _UNSPEAKABLE_LEN:
        return True
    s = _TICK_LIST_SEP.sub("` `", s)
    return any(not _NUMERIC_PATH.fullmatch(m) for m in _PATHLIKE.findall(s))


def _drain_think(buf: str):
    """Pull every complete thinking segment (newline- or sentence-
    terminated) off the front of buf, returning (segments, remainder).
    The trailing partial stays buffered for the next delta. Keeps
    reasoning streaming live rather than in one block-end lump."""
    out = []
    while True:
        m = _THINK_FLUSH.search(buf)
        if not m:
            break
        seg, buf = buf[:m.end()].strip(), buf[m.end():]
        if seg:
            out.append(seg)
    return out, buf


class _StreamSplitter:
    """Split a streamed text reply into spoken prose and fenced code.

    DISCIPLINE now lets the agent wrap code/config snippets in a
    ```` ``` ```` fence. Those must render on screen but never reach the
    mouth. feed() takes raw text_delta chunks and returns a list of
    complete prose sentences to speak; a fenced block (a line that
    strips to ```` ``` ```` opens it, the next such line closes it) is
    collected aside and handed to on_code the instant its closing fence
    lands, tagged for the transcript bus only.

    flush() forces out a trailing partial sentence at a block boundary
    (e.g. right before a tool call) without touching fence state.
    close() does the same at stream end and also releases an
    unterminated fence. `had_code` is True once any block was emitted
    this turn -- the caller uses it to decide whether a turn that spoke
    nothing still needs one "code's on screen" line so the face isn't
    silent."""

    def __init__(self, on_code):
        self._on_code = on_code
        self._raw = ""        # unprocessed text (may end mid-line)
        self._prose = ""      # prose awaiting sentence extraction
        self._code = ""       # body of the fence currently open
        self._in_fence = False
        self.had_code = False

    def feed(self, text):
        self._raw += text
        return self._pump(final=False, force_prose=False)

    def flush(self):
        return self._pump(final=False, force_prose=True)

    def close(self):
        return self._pump(final=True, force_prose=True)

    def _pump(self, final, force_prose):
        out = []
        while True:
            nl = self._raw.find("\n")
            if nl >= 0:
                line, self._raw = self._raw[:nl + 1], self._raw[nl + 1:]
            elif final and self._raw:
                line, self._raw = self._raw, ""
            else:
                break
            if line.strip().startswith("```"):
                if self._in_fence:
                    self._flush_code()
                    self._in_fence = False
                else:
                    out += self._drain_prose(force=True)
                    self._in_fence = True
                    self._code = ""
                continue
            if self._in_fence:
                self._code += line
            else:
                self._prose += line
                out += self._drain_prose(force=False)
        # whatever is left has no newline yet
        tail = self._raw
        if self._in_fence:
            if final:
                # stream ended inside a fence -- release what we have
                self._code += tail
                self._raw = ""
                self._flush_code()
                self._in_fence = False
            # else: code is not time-sensitive; keep the partial line
            # buffered so a closing fence split across deltas is spotted
        elif tail and not tail.lstrip().startswith("`"):
            # a partial line that cannot be a fence marker is safe to
            # speak now -- keeps the fast first-sentence start
            self._prose += tail
            self._raw = ""
            out += self._drain_prose(force=force_prose)
        elif final and tail:
            # stream ended on a line that looked like a fence but never
            # completed -- it was only prose
            self._prose += tail
            self._raw = ""
            out += self._drain_prose(force=True)
        elif force_prose:
            out += self._drain_prose(force=True)
        return out

    def _drain_prose(self, force):
        out = []
        while True:
            cut = _first_real_break(self._prose)
            if cut is None:
                break
            s = self._prose[:cut].strip()
            self._prose = self._prose[cut:]
            if s:
                self._route_prose(s, out)
        if force and self._prose.strip():
            self._route_prose(self._prose.strip(), out)
            self._prose = ""
        return out

    def _route_prose(self, s, out):
        """Speak s -- unless it is really a pasted block (a file path, a
        URL, a wall of text), in which case send it to the screen-only
        code role instead. Backstop for a model that ignores DISCIPLINE
        and narrates a file dump line by line."""
        if _looks_unspeakable(s):
            self.had_code = True
            self._on_code(s)
        else:
            out.append(s)

    def _flush_code(self):
        block = self._code.strip("\n")
        self._code = ""
        if block.strip():
            self.had_code = True
            self._on_code(block)


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")


def _squish(s, n) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _tool_summary(name: str, inp: dict) -> str:
    """One compact line describing a tool call, pushed to the transcript
    bus so a dashboard can show WHAT the agent is doing, not just what
    it's reasoning — the piece the desktop app's verbose view has that a
    plain chat panel doesn't. Never raises."""
    try:
        n = name or "?"
        inp = inp or {}
        if n == "Bash":
            return f"Bash: {_squish(inp.get('command', ''), 120)}"
        if n in ("Read", "Edit", "Write", "NotebookEdit"):
            p = str(inp.get("file_path") or inp.get("notebook_path") or "")
            return f"{n} {os.path.basename(p) or p}"
        if n == "Grep":
            where = inp.get("path") or inp.get("glob") or ""
            pat = _squish(inp.get("pattern", ""), 60)
            return f'Grep "{pat}"' + (f" in {where}" if where else "")
        if n == "Glob":
            return f"Glob {_squish(inp.get('pattern', ''), 80)}"
        if n in ("WebFetch", "WebSearch"):
            return f"{n} {_squish(inp.get('url') or inp.get('query') or '', 100)}"
        if n == "Task":
            return f"Task: {_squish(inp.get('description') or inp.get('subagent_type') or '', 80)}"
        if n == "TodoWrite":
            return "TodoWrite"
        for v in inp.values():
            if isinstance(v, (str, int, float)) and str(v).strip():
                return f"{n} {_squish(v, 90)}"
        return n
    except Exception:
        return name or "?"


def _tool_result_text(block) -> str:
    """Compact result string for the transcript bus (errors flagged).
    Never raises."""
    try:
        content = getattr(block, "content", None)
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    parts.append(str(c.get("text") or c.get("content") or ""))
                else:
                    parts.append(str(c))
            text = " ".join(parts)
        else:
            text = str(content or "")
        text = _squish(text, 200) or "(no output)"
        return ("error: " if getattr(block, "is_error", False) else "") + text
    except Exception:
        return ""


class WarmBrain:
    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None,
                 local_fallback: dict | None = None):
        # Full model id ON PURPOSE — never a bare alias. The SDK
        # resolves aliases through its own bundled CLI and can silently
        # land on an older model.
        self.model = model or CFG["model"]
        # What the session is actually on now — console /model switches
        # (deep / sonnet / default) move it; self.model stays the launch pick.
        self._live_model = self.model
        # The spoken permission gate (main.py builds it). Wired at
        # connect in EVERY mode, so a live mode flip needs no reconnect;
        # bypass simply never consults it.
        self._can_use_tool = can_use_tool
        # Session usage, spoken on request ("usage report").
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._client: ClaudeSDKClient | None = None
        # The session to reattach to at the FIRST start only (config key
        # resume_last_session). Consumed on use: a desync rebuild in
        # reset_turn() must always start FRESH: a rebuild means a turn
        # went sideways mid-stream, the wrong moment to gamble on
        # reattaching. (Community proposal, issue #1.)
        self._resume_id = resume_id
        # True while a query's response hasn't been consumed through its
        # ResultMessage — i.e. the shared message pipe may hold leftovers.
        self._dirty = False
        # Set per turn by ask_stream: did this turn emit a fenced code
        # block? main.speak_reply reads it to cover a turn that rendered
        # code but spoke nothing.
        self._turn_had_code = False
        # Consecutive _recover() calls with no completed turn between.
        # A resumed session replaying a poisoned oversized message dies
        # in the same spot forever (issue #38); after the first failed
        # resume, treat the saved session as poisoned and start fresh.
        self._recover_streak = 0
        # OPTIONAL break-glass local brain (config: local_fallback). None
        # unless enabled, and every hook below is an `if self._local`
        # guard that stays false -- the SDK path is byte-for-byte the
        # same behaviour when it's off. `local_fallback` as a parameter
        # exists for tests; main passes nothing and the config applies.
        lf = local_fallback if local_fallback is not None \
            else (CFG.get("local_fallback") or {})
        self._lf = lf
        self._local = None
        if lf.get("enabled"):
            # Lazy: local_brain imports this module for the splitter.
            from backtalk.local_brain import LocalBrain
            self._local = LocalBrain(
                lf, name=str(CFG.get("name") or "Assistant"),
                agent_dir=CFG["agent_dir"],
                extra_dirs=CFG.get("extra_dirs") or (),
                can_use_tool=can_use_tool)
        # True while turns are answered locally because Claude failed;
        # cleared the moment a cloud turn completes clean again.
        self.degraded = False
        self._degraded_turns = 0     # local turns since the switch (re-probe cadence)
        self._last_turn_local = False  # interrupt() must not poke the SDK for a local turn

    async def start(self):
        mode = CFG["permission_mode"]
        if mode == "default":
            mode = "ask"     # legacy alias, see config.py
        # backtalk's "ask" = the SDK's "default" mode with gated calls
        # routed to the spoken can_use_tool gate.
        sdk_mode = "default" if mode == "ask" else mode
        if sdk_mode == "bypassPermissions" and self._can_use_tool \
                and CanUseToolShadowedWarning:
            # Deliberate auto-approve: the SDK warns that the callback is
            # shadowed. That IS the chosen behavior, so boot quietly.
            warnings.filterwarnings("ignore",
                                    category=CanUseToolShadowedWarning)
        resume, self._resume_id = self._resume_id, None   # consume once

        def _opts(rid):
            return ClaudeAgentOptions(
                cwd=CFG["agent_dir"],
                model=self.model,
                system_prompt={"type": "preset", "preset": "claude_code",
                               "append": DISCIPLINE},
                include_partial_messages=True,
                # Without an explicit thinking config the CLI defaults
                # thinking.display to "omitted", so every transcript
                # thinking block is stored as signature-only with empty
                # text (SDK issue #831 — the caller has to opt in). The
                # model reasons either way; "summarized" is what makes
                # that reasoning readable, so nightly `dream` can scan a
                # backtalk session's thinking, not just its spoken turns.
                # "adaptive" (not "enabled" + budget_tokens): the model
                # paces its own reasoning depth per turn. budget_tokens is
                # removed on Sonnet 5 / Opus 5 (400 on the raw API) —
                # "adaptive" is the only on-mode; effort tunes depth.
                thinking={"type": "adaptive", "display": "summarized"},
                permission_mode=sdk_mode,
                can_use_tool=self._can_use_tool,
                hooks={"PostToolUse": [
                    HookMatcher(matcher="Bash", hooks=[_trim_bash_output]),
                ]},
                add_dirs=CFG["extra_dirs"],
                skills=CFG["visible_skills"],
                # The SDK frames the CLI's NDJSON one line per message and
                # aborts the whole stream ("Fatal error in message reader")
                # if any single line exceeds this. Default is 1 MB, which a
                # single attached photo blows straight through once Read
                # base64-encodes it into a tool_result. 50 MB clears one
                # max-size (~25 MB) /attach file with margin; the rare turn
                # that still overruns (several big images at once) is caught
                # by ask_stream's recovery path instead of bricking.
                max_buffer_size=50 * 1024 * 1024,
                resume=rid,
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                log(f"[brain] resumed session {resume[:8]}")
                return
            except Exception as e:
                # a stale or invalid saved session must never brick the
                # launch. Fall back to a fresh conversation and say so.
                log(f"[brain] resume failed ({str(e)[:80]}), "
                    f"starting fresh")
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
        self._client = ClaudeSDKClient(options=_opts(None))
        if not self._local:
            await self._client.connect()
            return
        # With a local brain configured, a connect that fails or hangs
        # (no sign-in, no network, no usage) degrades instead of killing
        # the launch -- main's boot guard would otherwise exit the whole
        # voice line. Re-raised only when the local brain can't come up
        # either, so that guard still speaks its line in that case.
        try:
            await asyncio.wait_for(self._client.connect(),
                                   _CONNECT_TIMEOUT_DEGRADABLE)
        except (Exception, asyncio.TimeoutError) as e:
            log(f"[brain] connect failed ({type(e).__name__}: "
                f"{str(e)[:120]})")
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            if not await self._local.ensure_up():
                raise
            self._enter_degraded()

    # ---- local fallback --------------------------------------------------

    def _enter_degraded(self):
        if self.degraded:
            return
        self.degraded = True
        self._degraded_turns = 0
        log("[brain] DEGRADED: Claude unreachable, answering on the local brain")
        signals.transcript("system", "Claude unreachable -- switched to the local brain")
        signals.set_session(degraded=True)

    async def _leave_degraded(self):
        self.degraded = False
        self._degraded_turns = 0
        log("[brain] recovered: Claude is answering again")
        signals.transcript("system", "Claude is back -- local brain released")
        signals.set_session(degraded=False)
        self._local.reset()
        if self._lf.get("stop_when_recovered", True):
            self._local.stop_server()

    async def _failover(self, utterance: str, why: str):
        """Answer THIS turn on the local brain, entering degraded mode
        (announced once) on the way in. Yields spoken lines."""
        log(f"[brain] failing over to local ({why})")
        if not await self._local.healthy():
            # first trigger of an outage: the server is started lazily,
            # and a cold model load is tens of seconds -- say so rather
            # than sit silent through it
            yield "Claude's not answering. One moment, waking the local backup brain."
            if not await self._local.ensure_up():
                yield ("It didn't come up, so I've got no brain to answer "
                       "with right now. Try me again in a bit.")
                return
        first = not self.degraded
        self._enter_degraded()
        if first:
            yield ("Okay, I'm on the local backup brain until Claude's "
                   "back. Slower and a good deal dumber: no memory, no "
                   "skills, just your files.")
        self._degraded_turns += 1
        self._last_turn_local = True
        self._dirty = True            # main's turn_active reads this
        try:
            async for s in self._local.ask_stream(utterance):
                yield s
        finally:
            self._turn_had_code = self._local._turn_had_code
            self._dirty = False       # no SDK pipe to drain for this turn

    async def set_permission_mode(self, backtalk_mode: str):
        """Live flip, no reconnect, conversation intact ("ask" maps to
        the SDK's "default", whose gated calls hit the spoken gate)."""
        if self._client:
            sdk_mode = "default" if backtalk_mode == "ask" \
                else backtalk_mode
            await self._client.set_permission_mode(sdk_mode)

    async def context_usage(self):
        """The CLI's own context-window breakdown, or None."""
        try:
            return await self._client.get_context_usage()
        except Exception:
            return None

    def _remember_session(self, rm):
        """Persist the session id after a completed turn, so the next
        launch can reattach (config: resume_last_session). Must never
        break a turn; silence on any failure."""
        if not CFG.get("resume_last_session"):
            return
        sid = getattr(rm, "session_id", None)
        if not sid:
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(sid)
        except OSError:
            pass

    def _tally(self, rm, count_turn=True):
        """Session usage bookkeeping. Must never break a turn."""
        try:
            u = getattr(rm, "usage", None) or {}
            s = self.session
            if count_turn:
                s["turns"] += 1
            s["out_tokens"] += int(u.get("output_tokens") or 0)
            s["in_tokens"] += (int(u.get("input_tokens") or 0)
                               + int(u.get("cache_read_input_tokens")
                                     or 0))
            c = getattr(rm, "total_cost_usd", None)
            if c:
                s["cost"] += float(c)
            # Spend, so gated exactly like the rate-limit readout.
            if CFG.get("show_usage"):
                signals.set_session(turns=s["turns"],
                                    cost=round(s["cost"], 4))
        except Exception:
            pass

    async def _pull_rate_limits(self):
        """Ask the CLI outright how much of the plan is spent.

        A DIRECT QUERY, not the RateLimitEvent stream. The event fires
        rarely and usually arrives carrying resets_at with no utilization
        at all, so a listener built on it reports nothing most of the
        time -- which is exactly how this feature looked broken for its
        whole life. (Community fix, ai-visualizer issue #1.)

        THIS REACHES PAST THE SDK'S PUBLIC SURFACE ON PURPOSE, and a
        reader should know it rather than discover it. `get_usage` is a
        control request the bundled CLI answers but the SDK never wraps,
        so there is no supported call to make. The supported-looking
        alternative is a dead end and was tested as one: the terminal
        status line never fires in a headless session, so its numbers
        are unreachable from here.

        Which means this can stop working without anyone doing anything
        wrong, and the containment is the point. Every failure is
        swallowed and the readout simply goes quiet. It must never cost
        a turn, so it is also bounded -- an unanswered control request
        would otherwise hang the voice line mid-conversation."""
        if not CFG.get("show_usage"):
            return
        try:
            usage = await asyncio.wait_for(
                self._client._query._send_control_request(
                    {"subtype": "get_usage"}), 5)
            for window in ("five_hour", "seven_day"):
                w = (usage.get("rate_limits") or {}).get(window)
                if not w:
                    continue
                # Two spellings accepted deliberately: this shape is not
                # documented anywhere, so the cheap tolerance is worth
                # more than the tidiness. Both are percentages, and the
                # rest of the pipeline wants a 0..1 fraction.
                pct = w.get("utilization")
                if pct is None:
                    pct = w.get("used_percentage")
                pct = pct / 100 if pct is not None else None
                resets = w.get("resets_at")
                if isinstance(resets, str):
                    resets = int(datetime.fromisoformat(resets).timestamp())
                signals.set_rate_limit(window, pct, resets)
        except Exception:
            pass

    async def _publish_context(self):
        """Publish context-window fill to the bus after a turn, so a
        dashboard can show it in its status row like the desktop app's
        context ring. Bounded and swallowed like _pull_rate_limits — it
        must never cost a turn. Unlike the rate-limit readout this is not
        gated: it is not account spend, only how full the window is."""
        try:
            cu = await asyncio.wait_for(self.context_usage(), 5)
            if not cu:
                return
            used = int(cu.get("totalTokens") or 0)
            total = int(cu.get("maxTokens") or cu.get("rawMaxTokens") or 0)
            if used or total:
                signals.set_context(used, total, cu.get("percentage"))
        except Exception:
            pass

    async def command(self, cmd: str) -> str:
        """Run a console slash command (/clear, /compact, /model,
        /effort) through the normal stream and return whatever text the
        CLI answered with (confirmations, errors). Slash-command replies
        arrive as COMPLETE AssistantMessages, not stream deltas, so
        ask_stream cannot see them. Bounded like reset_turn is: this
        stream is not trusted to always deliver, and an unbounded await
        here would deafen the whole voice loop. On timeout the pipe is
        left marked dirty so the next reset_turn drains or rebuilds."""
        if self._client is None:
            # only possible while degraded from boot: no CLI session exists
            return "error: Claude is unreachable, so console commands are off until it's back"
        self._dirty = True
        await self._client.query(cmd)
        texts = []

        async def _collect():
            async for msg in self._client.receive_response():
                t = type(msg).__name__
                if t == "AssistantMessage":
                    for b in getattr(msg, "content", []) or []:
                        txt = getattr(b, "text", None)
                        if txt:
                            texts.append(txt)
                elif t == "ResultMessage":
                    self._dirty = False
                    self._tally(msg, count_turn=False)
                    self._remember_session(msg)
                    break

        try:
            await asyncio.wait_for(_collect(), 90)
        except asyncio.TimeoutError:
            log(f"[brain] console command timed out: {cmd!r}")
            return "error: the command timed out"
        # /clear and /compact move the context floor a lot — refresh the
        # readout now rather than waiting for the next spoken turn.
        await self._publish_context()
        out = " ".join(texts).strip()
        low = out.lower()
        if not ("error" in low or "invalid" in low):
            if cmd.startswith("/model "):
                self._live_model = cmd.split(None, 1)[1].strip()
                signals.set_session(model=self._live_model)
            elif cmd.startswith("/effort "):
                signals.set_session(effort=cmd.split(None, 1)[1].strip())
        return out

    async def interrupt(self):
        # A cancelled local turn has nothing in the SDK pipe to stop.
        if self._client and not self._last_turn_local:
            await self._client.interrupt()

    async def reset_turn(self, timeout: float = 8.0):
        """Re-align the message pipe after an interrupted/failed turn.

        THE OFF-BY-ONE BUG, and why this method exists: the SDK client
        has ONE shared message stream and receive_response() stops at
        the FIRST ResultMessage it sees — there is no pairing between a
        query and its response. A cancelled turn stops consuming
        mid-stream, leaving the dead turn's remaining messages
        (including its ResultMessage) buffered. The next query then
        pairs with those leftovers: the first ask lands on the stale
        ResultMessage and yields nothing, and every ask after that
        answers the PREVIOUS question — for the rest of the session.
        So: interrupt the dead turn, then drain the pipe through its
        stale ResultMessage before the next query goes out. No-op when
        the last turn was consumed clean."""
        if not self._client or not self._dirty:
            return
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            pass  # turn may already be over — the drain below is the point

        async def _drain() -> int:
            n = 0
            async for msg in self._client.receive_response():
                n += 1
                if type(msg).__name__ == "ResultMessage":
                    break
            return n

        try:
            drained = await asyncio.wait_for(_drain(), timeout)
            log(f"[brain] interrupted turn drained ({drained} stale messages)")
            self._dirty = False
        except Exception:
            # Can't re-align — rebuild the session rather than run
            # desynced. Loses this voice session's conversation memory;
            # better than answering every question one turn late for the
            # rest of the day.
            log("[brain] stream desynced beyond repair — rebuilding the "
                "session (conversation memory for this session resets)")
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            await self.start()
            self._dirty = False

    async def stop(self):
        if self._client:
            await self._client.disconnect()
            self._client = None
        if self._local:
            if self._lf.get("stop_when_recovered", True):
                self._local.stop_server()   # only a server WE started
            await self._local.close()

    async def _recover(self, reason: str, quiet: bool = False):
        """A dead SDK client — transport crash mid-turn, most often a big
        tool result (an image Read, base64-encoded) overrunning the NDJSON
        line buffer — must not sit dead for the rest of the session. Tear
        it down and reconnect (resuming the last completed turn when one
        was saved, fresh otherwise), then speak one line so the turn isn't
        silent. Anything said since the last completed turn is lost — a
        working session beats a bricked one. On a second consecutive
        failure with no clean turn between, the saved session itself is
        the problem (it replays the killing message): move it aside and
        start fresh. (issue #38)

        quiet=True rebuilds without speaking: the caller is about to hand
        the turn to the local brain and will explain that instead."""
        log(f"[brain] recovering ({reason})")
        self._dirty = False
        try:
            await self._client.disconnect()
        except Exception:
            pass
        self._client = None
        self._recover_streak += 1
        # Second consecutive recovery with no clean turn between: the
        # saved session is replaying the message that kills us (issue
        # #38). Move it aside (not delete — recoverable by hand) and
        # start fresh instead of resuming into the same death.
        poisoned = self._recover_streak >= 2
        try:
            if CFG.get("resume_last_session"):
                if poisoned:
                    try:
                        os.replace(SESSION_FILE, f"{SESSION_FILE}.poisoned-"
                                   f"{datetime.now():%Y%m%d-%H%M%S}")
                        log("[brain] saved session poisoned — moved aside, "
                            "starting fresh")
                    except OSError:
                        pass
                    self._resume_id = None
                else:
                    try:
                        sid = open(SESSION_FILE).read().strip()
                    except OSError:
                        sid = ""
                    self._resume_id = sid or None
            await self.start()
            if quiet:
                return
            yield ("Something in our last session kept breaking my "
                   "connection, so I started fresh — tell me where we were."
                   if poisoned else
                   "I'm not getting anything back from my command-line "
                   "session — it may need to sign in again."
                   if reason == "TimeoutError" else
                   "Sorry — something you sent was too big and it dropped "
                   "my connection. I'm back now. Ask me again.")
        except Exception as e:
            log(f"[brain] recovery failed: {e!r}")
            if quiet:
                return
            yield ("I lost my connection and couldn't get it back. "
                   "Restart me when you get a chance.")

    async def ask_stream(self, utterance: str, _retried: bool = False):
        """Yield complete sentences as they stream out of the model."""
        first_timeout = _STREAM_FIRST_TIMEOUT
        if self._local:
            # Forced: every turn local (testing). Degraded: local, except
            # that every Nth turn probes Claude first and falls back
            # within the same turn if it's still down.
            n = int(self._lf.get("retry_cloud_every_n_turns") or 0)
            probe = (self.degraded and n > 0 and self._degraded_turns > 0
                     and self._degraded_turns % n == 0)
            if self._lf.get("force"):
                async for s in self._failover(utterance, "forced by config"):
                    yield s
                return
            if self.degraded and not probe:
                async for s in self._failover(utterance, "degraded"):
                    yield s
                return
            if probe:
                log("[brain] degraded: probing Claude this turn")
                first_timeout = _PROBE_FIRST_TIMEOUT
                if self._client is None:
                    # degraded since boot: no CLI session yet
                    try:
                        await self.start()
                    except Exception:
                        pass
                    if self._client is None:
                        async for s in self._failover(utterance, "probe: no session"):
                            yield s
                        return
        self._last_turn_local = False
        self._dirty = True             # in flight until its ResultMessage
        think_buf = ""                 # summarized reasoning, logged never spoken

        def _emit_code(block):
            log(f"[code] block ({len(block)} chars) -> screen only")
            signals.transcript("code", block)
            self._turn_had_code = True

        split = _StreamSplitter(_emit_code)
        self._turn_had_code = False
        _spoke_any = False
        _flagged = False               # safety classifier blocked this turn
        retry_on_sonnet = False
        try:
            await self._client.query(utterance)
            _it = self._client.receive_response().__aiter__()
            _first = True
            while True:
                # Bound ONLY the first message: no reply at all this long
                # after query() is a dead CLI subprocess (expired sign-in
                # is the field case). Once messages flow, a long gap is a
                # real tool call, so plain iteration from there.
                try:
                    if _first:
                        msg = await asyncio.wait_for(
                            _it.__anext__(), first_timeout)
                    else:
                        msg = await _it.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as te:
                    raise TimeoutError(
                        f"no first SDK message in {first_timeout}s "
                        "- the CLI session looks dead (expired sign-in?)"
                    ) from te
                _first = False
                t = type(msg).__name__
                if t == "StreamEvent":
                    ev = getattr(msg, "event", {}) or {}
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {}) or {}
                        if delta.get("type") == "text_delta":
                            # fence-aware: prose streams out to be spoken,
                            # fenced code is siphoned to the transcript bus
                            for sentence in split.feed(delta.get("text", "")):
                                _spoke_any = True
                                yield sentence
                        elif delta.get("type") == "thinking_delta":
                            # Reasoning stream: flush to the log AND the
                            # transcript bus AS IT FORMS (newline- or
                            # sentence-terminated segments), not in one lump
                            # at block end — a 12s reasoning pause otherwise
                            # shows nothing on screen then dumps the whole
                            # summary glued to the first spoken sentence.
                            # NEVER yielded — the mouth only ever speaks
                            # text_delta, so this stays screen-only, like the
                            # desktop app's verbose transcript. On the bus it
                            # rides as its own "thinking" role, distinct from
                            # user/assistant, so a dashboard can dim it.
                            think_buf += delta.get("thinking", "")
                            segs, think_buf = _drain_think(think_buf)
                            for seg in segs:
                                log(f"[think] {seg}")
                                signals.transcript("thinking", seg)
                    elif ev.get("type") == "content_block_stop":
                        if think_buf.strip():
                            seg = think_buf.strip()
                            log(f"[think] {seg}")
                            signals.transcript("thinking", seg)
                            think_buf = ""
                        # End of a speech block (e.g. right before a tool
                        # call): flush NOW. Without this, pre-tool filler
                        # ("On it — let me grab that.") sits silent in the
                        # buffer through the whole tool run, then plays
                        # glued to the answer: long dead air, then two
                        # thoughts at once. Fence state is left intact.
                        for tail in split.flush():
                            _spoke_any = True
                            yield tail
                elif t == "AssistantMessage":
                    # Tool calls: surface WHAT the agent does, not just its
                    # reasoning. Never yielded — transcript bus only, so a
                    # dashboard shows the activity like the desktop app's
                    # verbose view.
                    for b in getattr(msg, "content", []) or []:
                        if _SAFETY_FLAG in (getattr(b, "text", None) or ""):
                            _flagged = True
                        if type(b).__name__ in ("ToolUseBlock",
                                                "ServerToolUseBlock"):
                            line = _tool_summary(getattr(b, "name", "?"),
                                                 getattr(b, "input", {}))
                            log(f"[tool] {line}")
                            signals.transcript("tool", line)
                            # A tool is actually running now -- a distinct
                            # state from "thinking" so a face can show the
                            # turn isn't done just because the model stopped
                            # composing text.
                            signals.set_state("working")
                elif t == "UserMessage":
                    for b in getattr(msg, "content", []) or []:
                        if type(b).__name__ in ("ToolResultBlock",
                                                "ServerToolResultBlock"):
                            res = _tool_result_text(b)
                            if res:
                                signals.transcript("tool-result", res)
                            # Tool done -- the model goes back to reasoning
                            # about the result before it speaks or calls the
                            # next tool.
                            signals.set_state("thinking")
                elif t == "ResultMessage":
                    self._dirty = False   # turn fully consumed — pipe aligned
                    self._recover_streak = 0   # clean turn — not stuck
                    self._tally(msg)
                    self._remember_session(msg)
                    await self._pull_rate_limits()
                    await self._publish_context()
                    _flagged = (_flagged
                                or getattr(msg, "stop_reason", None) == "refusal"
                                or _SAFETY_FLAG in (getattr(msg, "result", None)
                                                    or ""))
                    if _flagged:
                        log(f"[brain] safety classifier blocked the turn "
                            f"(model={self._live_model}, stop_reason="
                            f"{getattr(msg, 'stop_reason', None)})")
                        for tail in split.flush():
                            _spoke_any = True
                            yield tail
                        if (not _retried
                                and self._live_model != CFG["sonnet_model"]):
                            retry_on_sonnet = True
                        else:
                            yield ("The safety filter flagged that again. "
                                   "Say clear the session to start fresh.")
                    elif getattr(msg, "is_error", False):
                        errs = "; ".join(getattr(msg, "errors", None) or [])
                        status = getattr(msg, "api_error_status", None)
                        log("[brain] turn returned an error"
                            + (f" (HTTP {status})" if status else "")
                            + (f": {errs[:200]}" if errs else ""))
                        for tail in split.flush():
                            _spoke_any = True
                            yield tail
                        if self._local and _is_outage(status, errs):
                            # TRIGGER 1: the API itself is down, limited,
                            # or unreachable -> this turn goes local.
                            async for line in self._failover(
                                    utterance, f"HTTP {status}" if status
                                    else (errs[:60] or "error result")):
                                yield line
                            break
                        yield ("Something went wrong on my end and the "
                               "turn came back empty. If this keeps up, "
                               "my command-line session may need to sign "
                               "in again.")
                    elif not _spoke_any and not self._turn_had_code:
                        for tail in split.flush():
                            _spoke_any = True
                            yield tail
                        if not _spoke_any:
                            yield "I got nothing back that time. Ask me again."
                    if self._local and self.degraded \
                            and not getattr(msg, "is_error", False):
                        # a clean cloud turn while degraded = the probe
                        # succeeded: Claude answered this one, release
                        # the local brain
                        await self._leave_degraded()
                        yield "And Claude's back, so I'm off the local brain."
                    break
        except GeneratorExit:
            raise
        except Exception as e:
            # SDK transport died mid-turn — most often CLIJSONDecodeError,
            # a tool result (an image Read, base64-encoded) overrunning
            # max_buffer_size. Flush whatever prose we have, rebuild the
            # client so the NEXT turn works, and say what happened.
            log(f"[brain] stream died: {type(e).__name__}: {str(e)[:160]}")
            for tail in split.flush():
                yield tail
            # TRIGGER 2: no first message at all = the CLI can't reach
            # the API (network gone, sign-in dead). Rebuild the session
            # quietly and answer this turn locally. Any other transport
            # death (an oversized tool result) is not an outage and keeps
            # the spoken recovery exactly as before.
            outage = self._local is not None and isinstance(e, TimeoutError)
            async for line in self._recover(type(e).__name__, quiet=outage):
                yield line
            if outage:
                async for line in self._failover(utterance, "TimeoutError"):
                    yield line
            return
        for tail in split.close():
            yield tail
        if retry_on_sonnet:
            yield ("Opus's safety filter flagged that, probably a false "
                   "positive. Switching to Sonnet for this session and "
                   "trying again.")
            resp = (await self.command(f"/model {CFG['sonnet_model']}")).lower()
            if "error" in resp or "invalid" in resp:
                log(f"[brain] auto-switch to Sonnet failed: {resp[:120]}")
                yield ("I couldn't switch to Sonnet. Say clear the session "
                       "to start fresh.")
                return
            await self.command("/effort high")
            async for s in self.ask_stream(utterance, _retried=True):
                yield s


if __name__ == "__main__":
    import time

    async def demo():
        b = WarmBrain()
        await b.start()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            t0 = time.time()
            async for s in b.ask_stream(prompt):
                print(f"  ({time.time()-t0:4.1f}s) {s}", flush=True)
        await b.stop()

    asyncio.run(demo())
