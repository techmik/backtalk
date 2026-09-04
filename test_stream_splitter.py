"""Focused tests for brain._StreamSplitter: prose is yielded to be
spoken, fenced code is siphoned to the on_code callback and never
yielded, and both survive being fed one character at a time."""
from backtalk.brain import _StreamSplitter


def run(chunks, *, char_by_char=False):
    """Feed chunks (optionally exploded to single chars), return
    (spoken_sentences, code_blocks)."""
    blocks = []
    s = _StreamSplitter(blocks.append)
    spoken = []
    feed = []
    for c in chunks:
        feed.extend(c) if char_by_char else feed.append(c)
    for piece in feed:
        spoken += s.feed(piece)
    spoken += s.close()
    return spoken, blocks, s


def test_plain_prose_unchanged():
    spoken, blocks, _ = run(["Hello there. ", "How are you? ", "Fine."])
    assert spoken == ["Hello there.", "How are you?", "Fine."]
    assert blocks == []


def test_simple_fence_diverted():
    spoken, blocks, s = run([
        "Here is the config. ",
        "```\n",
        "key = value\n",
        "other = 1\n",
        "```\n",
        "That sets two keys.",
    ])
    assert spoken == ["Here is the config.", "That sets two keys."]
    assert blocks == ["key = value\nother = 1"]
    assert s.had_code is True


def test_fence_with_language_tag():
    spoken, blocks, _ = run([
        "Try this.\n```python\nx = 1\n```\nDone.",
    ])
    assert spoken == ["Try this.", "Done."]
    assert blocks == ["x = 1"]


def test_closing_fence_split_across_deltas():
    spoken, blocks, _ = run([
        "Look.\n``", "`\n", "a = 1\n", "`", "`", "`", "\n", "Ok now."
    ])
    assert spoken == ["Look.", "Ok now."]
    assert blocks == ["a = 1"]


def test_char_by_char_matches():
    chunks = ["Intro line here. ",
              "```\ndef f():\n    return 2\n```\n",
              "That is a function. It returns two."]
    a = run(chunks)
    b = run(chunks, char_by_char=True)
    assert a[0] == b[0] == ["Intro line here.",
                            "That is a function.", "It returns two."]
    assert a[1] == b[1] == ["def f():\n    return 2"]


def test_unterminated_fence_flushed_at_close():
    spoken, blocks, s = run(["Here.\n```\nhalf a block\nno closer"])
    assert spoken == ["Here."]
    assert blocks == ["half a block\nno closer"]
    assert s.had_code is True


def test_code_only_turn_sets_flag_no_speech():
    spoken, blocks, s = run(["```\njust code\n```\n"])
    assert spoken == []
    assert blocks == ["just code"]
    assert s.had_code is True


def test_flush_releases_partial_prose_keeps_fence_state():
    blocks = []
    s = _StreamSplitter(blocks.append)
    out = s.feed("Partial thought with no period")
    assert out == []
    out = s.flush()                      # e.g. right before a tool call
    assert out == ["Partial thought with no period"]
    # a fence opened after the flush still works
    out = s.feed("\n```\ncode\n```\n")
    out += s.close()
    assert blocks == ["code"]


def test_prose_between_two_blocks():
    spoken, blocks, _ = run([
        "First.\n```\nA\n```\nMiddle sentence here.\n```\nB\n```\nLast.",
    ])
    assert spoken == ["First.", "Middle sentence here.", "Last."]
    assert blocks == ["A", "B"]


def test_inline_backticks_are_not_a_fence():
    spoken, blocks, _ = run(["Set the `foo` flag to true please."])
    assert spoken == ["Set the `foo` flag to true please."]
    assert blocks == []


def test_pathspam_sentence_rerouted_to_screen():
    # Model ignored DISCIPLINE and narrated a file dump -- the backstop
    # sends the path-bearing line to the screen-only code role.
    spoken, blocks, s = run([
        "Here are the changes. ",
        "Edit C:\\Users\\MikeS\\Documents\\MyJarvis\\.claude\\skills\\dream\\SKILL.md line ten. ",
        "That's it.",
    ])
    assert spoken == ["Here are the changes.", "That's it."]
    assert len(blocks) == 1 and "SKILL.md" in blocks[0]
    assert s.had_code is True


def test_forward_slash_path_rerouted():
    spoken, blocks, _ = run(["Open src/tts/playback.py and check the rate."])
    assert spoken == []
    assert blocks == ["Open src/tts/playback.py and check the rate."]


def test_bare_filename_still_spoken():
    spoken, blocks, _ = run(["Check ears.py for the mic setting please."])
    assert spoken == ["Check ears.py for the mic setting please."]
    assert blocks == []


def test_single_separator_still_spoken():
    spoken, blocks, _ = run(["Put it in the src/ folder for now."])
    assert spoken == ["Put it in the src/ folder for now."]
    assert blocks == []


def test_date_and_ratio_still_spoken():
    # digits-and-separators only is a date or a ratio, not a path
    # (the daily recap's calendar line was being diverted; 2026-09-04)
    spoken, blocks, _ = run(["Your appointment is 9/22/2026 at nine. ",
                             "Support runs 24/7/365."])
    assert spoken == ["Your appointment is 9/22/2026 at nine.",
                      "Support runs 24/7/365."]
    assert blocks == []


def test_lettered_multi_segment_path_still_rerouted():
    spoken, blocks, _ = run(["Edit 03 - Areas/Health/CVS Treatment History.md now."])
    assert spoken == []
    assert len(blocks) == 1


def test_overlong_sentence_rerouted():
    wall = "word " * 90 + "end."          # ~450 chars, no path
    spoken, blocks, s = run([wall])
    assert spoken == []
    assert blocks == [wall.strip()]
    assert s.had_code is True


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception:
            bad += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
