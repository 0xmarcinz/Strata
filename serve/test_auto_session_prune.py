"""serve/test_auto_session_prune.py - the age rule on the parked conversations, and what it must never touch.

`auto_sessions_max_age_days` deletes `auto-<16 hex>.bin` files older than the configured days.  Everything here is
about the "never": a symbolic link (not followed, not deleted), a directory with the same name, a file saved by hand
through /slots/0, a temporary file, an uppercase or truncated name, the conversation the engine holds right now, a
file written seconds ago (a save may be in flight) and a file that was replaced between the scan and the delete.
Nothing in a request or a page reaches these decisions: the names come from the folder itself.

    python -m unittest serve.test_auto_session_prune -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import AUTO_SESSION_FILE, AUTO_SESSION_SAVE_GRACE, Service  # noqa: E402

OLD = 4 * 86400          # a file four days old
HASH = "0123456789abcdef"


def name(key: str = HASH) -> str:
    return f"auto-{key}.bin"


def service(dirpath, *, keep=0, max_gb=0.0, age_days=0.0, held=None, request_key=None):
    """The service's pruner state only (Service.__new__: no engine, no GPU, no HTTP)."""
    svc = Service.__new__(Service)
    svc.slot_save_path = str(dirpath)
    svc.auto_sessions_keep = keep
    svc.auto_sessions_max_gb = max_gb
    svc.auto_sessions_max_age_days = age_days
    svc.auto_conv = {"key": held, "tokens": 1234} if held else None
    svc.auto_tls = threading.local()
    if request_key:
        svc.auto_tls.key = request_key
    svc._disk_cache = (0.0, None)
    return svc


class Prune(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.outside = Path(self._tmp.name + "-outside")
        self.outside.mkdir(exist_ok=True)

    def tearDown(self):
        for p in self.dir.iterdir():
            try:
                if p.is_symlink() or p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except OSError:
                pass
        self._tmp.cleanup()
        for p in self.outside.iterdir():
            p.unlink()
        self.outside.rmdir()

    def make(self, filename, *, age=OLD, size=10):
        p = self.dir / filename
        p.write_bytes(b"x" * size)
        t = __import__("time").time() - age
        os.utime(p, (t, t))
        return p

    # ---- the age rule itself -------------------------------------------------

    def test_old_files_go_and_young_ones_stay(self):
        old = self.make(name(), age=OLD)
        young = self.make(name("fedcba9876543210"), age=60)
        report = service(self.dir, age_days=1)._auto_session_prune()
        self.assertEqual((report["removed"], report["files"]), (1, 1))
        self.assertFalse(old.exists())
        self.assertTrue(young.exists())

    def test_no_age_rule_keeps_everything(self):
        old = self.make(name(), age=OLD)
        service(self.dir, age_days=0)._auto_session_prune()
        self.assertTrue(old.exists())

    def test_a_future_timestamp_is_not_old(self):
        future = self.make(name(), age=-86400)                # clock skew: mtime in the future
        service(self.dir, age_days=1)._auto_session_prune()
        self.assertTrue(future.exists())

    def test_a_file_written_moments_ago_is_left_alone(self):
        fresh = self.make(name(), age=1)                       # inside the save grace: a save may be in flight
        service(self.dir, age_days=0.000001)._auto_session_prune()
        self.assertTrue(fresh.exists())
        self.assertGreater(AUTO_SESSION_SAVE_GRACE, 1)

    # ---- names that are not ours --------------------------------------------

    def test_only_auto_hash_bin_is_ever_deleted(self):
        keep = [self.make("chat1.bin"),                       # saved by hand through /slots/0
                self.make("auto-.bin"),                       # no key
                self.make("auto-0123.bin"),                   # too short
                self.make("auto-0123456789ABCDEF.bin"),       # uppercase hex is not a key we write
                self.make("auto-0123456789abcdef.bin.tmp"),   # a temporary file
                self.make("AUTO-0123456789abcdef.bin"),       # wrong prefix
                self.make("prefix-0123456789abcdef.bin")]     # a shared start: never aged out
        doomed = self.make(name(), age=OLD)
        service(self.dir, age_days=1)._auto_session_prune()
        self.assertFalse(doomed.exists())
        for p in keep:
            self.assertTrue(p.exists(), p.name)

    def test_the_pattern_is_exact(self):
        for good in (name(), "auto-0000000000000000.bin"):
            self.assertTrue(AUTO_SESSION_FILE.match(good), good)
        for bad in ("auto-0123456789abcde.bin", "auto-0123456789abcdefg.bin", "auto-0123456789abcdef.bin\n",
                    "../auto-0123456789abcdef.bin", "auto-0123456789abcdef.bin/x", "auto-0123456789abcdef.BIN"):
            self.assertFalse(AUTO_SESSION_FILE.match(bad), bad)

    # ---- links and other file types -----------------------------------------

    def test_a_symlink_is_not_followed_and_not_deleted(self):
        target = self.outside / "precious.bin"
        target.write_bytes(b"do not touch")
        t = __import__("time").time() - OLD
        os.utime(target, (t, t))
        link = self.dir / name()
        link.symlink_to(target)
        service(self.dir, age_days=1)._auto_session_prune()
        self.assertTrue(link.is_symlink())                    # the link itself stays
        self.assertTrue(target.exists())                      # and its target was never opened or unlinked
        self.assertEqual(target.read_bytes(), b"do not touch")

    def test_a_directory_with_that_name_is_skipped(self):
        d = self.dir / name()
        d.mkdir()
        t = __import__("time").time() - OLD
        os.utime(d, (t, t))
        service(self.dir, age_days=1)._auto_session_prune()
        self.assertTrue(d.is_dir())

    def test_a_hardlink_is_removed_but_the_other_name_survives(self):
        target = self.outside / "data.bin"
        target.write_bytes(b"shared inode")
        t = __import__("time").time() - OLD
        os.utime(target, (t, t))
        os.link(target, self.dir / name())
        service(self.dir, age_days=1)._auto_session_prune()
        self.assertFalse((self.dir / name()).exists())
        self.assertTrue(target.exists(), "only the name inside the folder may go")

    # ---- the conversations the server is holding ----------------------------

    def test_the_held_and_the_current_conversation_are_never_deleted(self):
        held = self.make(name("aaaaaaaaaaaaaaa1"), age=OLD)
        current = self.make(name("bbbbbbbbbbbbbbb2"), age=OLD)
        other = self.make(name("ccccccccccccccc3"), age=OLD)
        service(self.dir, age_days=1, held="aaaaaaaaaaaaaaa1", request_key="bbbbbbbbbbbbbbb2")._auto_session_prune()
        self.assertTrue(held.exists())
        self.assertTrue(current.exists())
        self.assertFalse(other.exists())

    # ---- the existing limits keep working -----------------------------------

    def test_count_limit_still_deletes_the_oldest(self):
        oldest = self.make(name("aaaaaaaaaaaaaaa1"), age=3 * 86400)
        middle = self.make(name("bbbbbbbbbbbbbbb2"), age=2 * 86400)
        newest = self.make(name("ccccccccccccccc3"), age=86400)
        service(self.dir, keep=2)._auto_session_prune()
        self.assertFalse(oldest.exists())
        self.assertTrue(middle.exists())
        self.assertTrue(newest.exists())

    def test_size_cap_still_deletes_the_oldest(self):
        old = self.make(name("aaaaaaaaaaaaaaa1"), age=3 * 86400, size=2 << 20)
        new = self.make(name("bbbbbbbbbbbbbbb2"), age=86400, size=2 << 20)
        service(self.dir, max_gb=(3 << 20) / (1 << 30))._auto_session_prune()   # a 3 MiB cap, 4 MiB present
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    # ---- a file replaced between the scan and the delete --------------------

    def test_a_file_replaced_after_the_scan_is_left_alone(self):
        victim = self.make(name(), age=OLD)
        svc = service(self.dir, age_days=1)
        stale = svc._auto_session_files()                     # what the pruner saw
        victim.unlink()
        victim.write_bytes(b"a brand new file")               # same name, different inode (e.g. an in-place save)
        svc._auto_session_files = lambda: stale               # the pruner still holds the old inode
        svc._auto_session_prune()
        self.assertTrue(victim.exists(), "an inode that no longer matches the scan must not be unlinked")
        self.assertEqual(victim.read_bytes(), b"a brand new file")


if __name__ == "__main__":
    unittest.main()
