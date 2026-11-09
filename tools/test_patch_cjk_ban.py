"""tools/test_patch_cjk_ban.py - the fork patcher: known layout, idempotence, atomicity, no blind writes."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from tools.patch_cjk_ban import CHANGES, IMPORT, CONFIG, MARKER, PatchError, install, patch_server  # noqa: E402


class ForkPatch(unittest.TestCase):
    def original_server(self) -> str:
        """This tree's server.py with the hook lines taken back out: what a fork looks like before the patch."""
        lines = (ROOT / "serve" / "server.py").read_text(encoding="utf-8").splitlines()
        return "\n".join(line for line in lines if MARKER not in line)

    def test_this_tree_carries_both_anchors(self):
        source = self.original_server()
        for old, _ in CHANGES:
            self.assertEqual(source.count(old), 1, old[:60])

    def test_this_tree_is_patchable(self):
        original = self.original_server()
        patched = patch_server(original)
        self.assertEqual(patched.count(MARKER), 2)
        self.assertIn(IMPORT, patched)
        self.assertIn(CONFIG, patched)

    def test_complete_hook_is_idempotent(self):
        patched = patch_server(self.original_server())
        self.assertEqual(patch_server(patched), patched)

    def test_unknown_or_partial_source_is_rejected(self):
        for source in ["class Service: pass", self.original_server() + "\n" + MARKER]:
            with self.subTest(source=source[:20]), self.assertRaises(PatchError):
                patch_server(source)

    def test_install_writes_both_files_and_a_second_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            server = Path(d) / "server.py"
            server.write_text(self.original_server(), encoding="utf-8")
            first = install(server)
            self.assertEqual(first["changed"], 2)                     # the server and cjk_ban.py
            self.assertTrue((Path(d) / "cjk_ban.py").is_file())
            self.assertEqual(first["bias_memo"], "absent")            # no logit_bias.py beside it: only a note
            before = {p.name: p.read_bytes() for p in Path(d).glob("*.py")}
            second = install(server)
            self.assertEqual(second["changed"], 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(d).glob("*.py")})

    def test_a_missing_anchor_leaves_every_file_alone(self):
        with tempfile.TemporaryDirectory() as d:
            server = Path(d) / "server.py"
            server.write_text("import os\nRELEASE = 1\n", encoding="utf-8")
            before = server.read_bytes()
            with self.assertRaises(PatchError):
                install(server)
            self.assertEqual(server.read_bytes(), before)
            self.assertFalse((Path(d) / "cjk_ban.py").exists())

    def test_the_fork_bias_module_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            server = Path(d) / "server.py"
            server.write_text(self.original_server(), encoding="utf-8")
            (Path(d) / "logit_bias.py").write_text("def normalize(v, n=None):\n    return {}\n", encoding="utf-8")
            mark = (Path(d) / "logit_bias.py").read_bytes()
            result = install(server)
            self.assertEqual(result["bias_memo"], "no")               # no memo: reported, not rewritten
            self.assertEqual((Path(d) / "logit_bias.py").read_bytes(), mark)

    def test_crlf_sources_keep_their_line_endings(self):
        with tempfile.TemporaryDirectory() as d:
            server = Path(d) / "server.py"
            server.write_bytes(self.original_server().replace("\n", "\r\n").encode("utf-8"))
            install(server)
            written = server.read_bytes()
            self.assertIn(b"\r\n" + CONFIG.encode("utf-8"), written)
            self.assertNotIn(b"\n" + CONFIG.encode("utf-8") + b"\r\n", written.replace(b"\r\n", b"\x00"))


if __name__ == "__main__":
    unittest.main()
