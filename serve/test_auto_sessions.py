"""serve/test_auto_sessions.py - the order of the disk-parking steps around a conversation switch (#3565).

The engine holds one conversation.  When the next request starts another one, the held conversation is saved to
its file first, and only then may something restore over the engine's session: this request's own file, or - when
it has none - its shared start (`prefix_sessions`).  A restore that runs first writes the RESTORED state into the
outgoing conversation's file; the next time that conversation comes back it restores a state that is not its own
and reads its whole prompt again.  That is what an orchestrator and its sub-agents did on a live server: an
orchestrator's and a sub-agent's shared starts differ, so each switch restored the other's prefix as its history
(`restored 16689 tokens`, then `reading the prompt: 36,207 of 36,212`, reused 0).

    python -m unittest serve.test_auto_sessions -v
"""
from __future__ import annotations

import collections
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import ByteTokenizer, Service  # noqa: E402

A, B, P = "aaaaaaaaaaaaaaa1", "bbbbbbbbbbbbbbb2", "ccccccccccccccc3"   # two conversations and one shared start


class FakeEngine:
    """Records the SAVE/RESTORE order; `tokens` is what the engine reports for the state it just moved."""

    def __init__(self, tokens=(1000, 2000)):
        self.calls: list[tuple[str, str]] = []
        self.tokens = list(tokens)

    def session_file(self, action, path):
        self.calls.append((action, os.path.basename(path)))
        if action == "save":
            Path(path).write_bytes(b"x")              # the engine writes the file; a restore needs one to find
        n = self.tokens[0] if action == "save" else self.tokens[-1]
        return {"tokens": n, "bytes": 1024, "ms": 1.0}


def service(dirpath: str, engine) -> Service:
    """The service's state for one switch (Service.__new__: no engine start, no GPU, no HTTP)."""
    svc = Service.__new__(Service)
    svc.engine = engine
    svc.tok = ByteTokenizer()
    svc.auto_tls = threading.local()
    svc.auto_conv = None
    svc.auto_restored = 0
    svc.auto_prompts = collections.OrderedDict()
    svc.prefix_pending = None
    svc.prefix_applied = None
    svc.prefix_sessions = True
    svc.slot_save_path = dirpath
    svc.auto_sessions = True
    svc.auto_sessions_min_tokens = 100
    svc.auto_sessions_keep = 1000
    svc.auto_sessions_max_gb = 10.0
    svc.auto_stats = {}
    svc.prefix_stats = {}
    svc._disk_cache = (0.0, None)
    return svc


class AutoSessionOrderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.engine = FakeEngine()

    def tearDown(self):
        self.tmp.cleanup()

    def turn(self, svc, key, prefix, n=5000):
        svc.auto_tls.key, svc.auto_tls.prefix = key, (prefix, n)
        svc._auto_session_turn([1] * n)
        return self.engine.calls

    def test_held_conversation_is_saved_before_the_shared_start_restores(self):
        svc = service(self.dir, self.engine)
        svc.auto_conv = {"key": A, "tokens": 5000}
        Path(self.dir, f"prefix-{P}.bin").write_bytes(b"x")     # a shared start on disk
        calls = self.turn(svc, B, P)                            # B has no file of its own
        self.assertEqual(calls, [("save", f"auto-{A}.bin"), ("restore", f"prefix-{P}.bin")])

    def test_own_file_wins_over_the_shared_start(self):
        svc = service(self.dir, self.engine)
        svc.auto_conv = {"key": A, "tokens": 5000}
        Path(self.dir, f"prefix-{P}.bin").write_bytes(b"x")
        Path(self.dir, f"auto-{B}.bin").write_bytes(b"x")
        calls = self.turn(svc, B, P)
        self.assertEqual(calls, [("save", f"auto-{A}.bin"), ("restore", f"auto-{B}.bin")])

    def test_a_newer_shared_start_never_replaces_a_restored_conversation(self):
        svc = service(self.dir, self.engine)
        svc.auto_conv = {"key": A, "tokens": 5000}
        svc.prefix_applied = "differentshared"                  # another agent's start: it would restore again
        for name in (f"prefix-{P}.bin", f"auto-{B}.bin"):
            Path(self.dir, name).write_bytes(b"x")
        calls = self.turn(svc, B, P)
        self.assertEqual([a for a, _ in calls], ["save", "restore"])
        self.assertEqual(calls[-1], ("restore", f"auto-{B}.bin"))
        self.assertNotIn(("restore", f"prefix-{P}.bin"), calls)

    def test_the_previous_requests_shared_start_is_written_while_the_engine_still_holds_it(self):
        svc = service(self.dir, self.engine)
        q = Path(self.dir, f"prefix-{P}.bin")
        svc.prefix_pending = (str(q), 5000)                     # the previous request carried P, no file yet
        svc.auto_conv = {"key": A, "tokens": 5000}
        Path(self.dir, f"auto-{B}.bin").write_bytes(b"x")
        calls = self.turn(svc, B, P)
        self.assertEqual(calls[0], ("save", f"prefix-{P}.bin"))
        self.assertEqual(calls[1], ("save", f"auto-{A}.bin"))
        self.assertTrue(q.is_file())

    def test_same_conversation_is_left_alone(self):
        svc = service(self.dir, self.engine)
        svc.auto_conv = {"key": A, "tokens": 5000}
        Path(self.dir, f"auto-{A}.bin").write_bytes(b"x")
        self.assertEqual(self.turn(svc, A, P, n=6000), [])
        self.assertEqual(svc.auto_conv["tokens"], 6000)

    def test_a_short_held_conversation_is_not_saved_but_the_incoming_one_still_restores(self):
        svc = service(self.dir, self.engine)
        svc.auto_conv = {"key": A, "tokens": 50}                 # below auto_sessions_min_tokens
        Path(self.dir, f"auto-{B}.bin").write_bytes(b"x")
        self.assertEqual(self.turn(svc, B, P), [("restore", f"auto-{B}.bin")])


if __name__ == "__main__":
    unittest.main()


class PartialMatchTest(unittest.TestCase):
    """A restore that the prompt only partly matches says so, with the point where they part (the engine's reuse)."""

    def test_a_rewritten_prompt_is_reported_with_its_divergence_point(self):
        svc = service(tempfile.mkdtemp(), FakeEngine())
        note = svc._auto_session_check(41434, 15085, 41467)          # the miss measured on 2026-11-02 12:47
        self.assertIn("41434", note)
        self.assertIn("after 15085", note)
        self.assertIn("26382", note)

    def test_a_state_used_whole_says_nothing(self):
        svc = service(tempfile.mkdtemp(), FakeEngine())
        self.assertIsNone(svc._auto_session_check(41434, 41419, 41467))
        self.assertIsNone(svc._auto_session_check(41434, 41434, 41467))

    def test_nothing_restored_or_no_engine_figure_says_nothing(self):
        svc = service(tempfile.mkdtemp(), FakeEngine())
        self.assertIsNone(svc._auto_session_check(0, 100, 500))
        self.assertIsNone(svc._auto_session_check(41434, None, 41467))


class PromptMovedTest(unittest.TestCase):
    """Where a prompt differs from the previous turn of its own conversation: inside the shared start (the system
    prompt or the tool list) or after it (the history).  That is the difference between "the client changed its
    instructions" and "the client rewrote its history" - both make the engine read the rest again."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.svc = service(self.dir, FakeEngine())
        self.svc.auto_prompts = collections.OrderedDict()

    def turn(self, key, ids, shared=0):
        self.svc.auto_tls.key = key
        return self.svc._auto_prompt_moved(ids, shared)

    def test_an_ordinary_turn_says_nothing(self):
        self.assertIsNone(self.turn("k", [1, 2, 3], shared=2))
        self.assertIsNone(self.turn("k", [1, 2, 3, 4, 5], shared=2))

    def test_a_change_inside_the_shared_start_names_the_system_prompt_or_tools(self):
        self.turn("k", [1, 2, 3, 4, 5], shared=3)
        note = self.turn("k", [1, 2, 9, 4, 5], shared=3)
        self.assertIn("shared start changed at token 2", note)
        self.assertIn("tool list", note)

    def test_a_change_after_the_shared_start_names_the_history(self):
        self.turn("k", [1, 2, 3, 4, 5, 6], shared=2)
        note = self.turn("k", [1, 2, 3, 9, 5, 6, 7], shared=2)
        self.assertIn("history changed at token 3 of the previous turn's 6", note)

    def test_the_first_turn_of_a_conversation_says_nothing(self):
        self.assertIsNone(self.turn("new", [1, 2, 3]))

    def test_conversations_are_compared_with_their_own_previous_turn(self):
        self.turn("a", [1, 1, 1], shared=3)
        self.turn("b", [2, 2, 2, 7], shared=3)           # interleaved: b must not be compared with a
        self.assertIsNone(self.turn("a", [1, 1, 1, 1], shared=3))
        self.assertIn("history changed at token 3", self.turn("b", [2, 2, 2, 9], shared=3))

    def test_only_the_eight_most_recent_conversations_are_kept(self):
        for i in range(9):
            self.turn(f"k{i}", [i, i, i])
        self.assertEqual(len(self.svc.auto_prompts), 8)
        self.assertIsNone(self.turn("k0", [0, 0, 0]))     # forgotten: nothing to compare, no complaint


class ExcerptTest(unittest.TestCase):
    """The log shows the text either side of the divergence, so the client's change is recognisable from here."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.svc = service(self.dir, FakeEngine())

    def ids(self, text):
        return self.svc.tok.encode(text)

    def test_the_two_prompts_are_shown_where_they_part(self):
        svc = self.svc
        svc.auto_prompts = collections.OrderedDict()
        svc.auto_tls.key = "k"
        svc._auto_prompt_moved(self.ids("system prompt\nstep 4: read the file\nanswer so far"), 0)
        note = svc._auto_prompt_moved(self.ids("system prompt\nstep 5: read the file\nanswer so far"), 0)
        self.assertIn("was:", note)
        self.assertIn("now:", note)
        self.assertIn("step 4", note)
        self.assertIn("step 5", note)

    def test_an_undecodable_window_does_not_fail_the_request(self):
        svc = self.svc
        svc.auto_prompts = collections.OrderedDict()
        svc.auto_tls.key = "k"
        svc._auto_prompt_moved([256, 257, 258], 0)      # specials only
        note = svc._auto_prompt_moved([256, 259, 258], 0)
        self.assertIn("was:", note)                     # whatever it decodes to, it is a line, not an exception
