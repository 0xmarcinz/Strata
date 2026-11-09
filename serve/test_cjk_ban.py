"""serve/test_cjk_ban.py - the opt-in Han default ban (`ban_cjk_tokens`), and the hook that installs it."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from serve.cjk_ban import (BYTE_TO_UNICODE, UNICODE_TO_BYTE, configure_cjk_ban, contains_han,  # noqa: E402
                           han_token_ids, raw_surface, surface)
from tools.strata_tokenizer import UNICODE_TO_BYTE as ORACLE_UNICODE_TO_BYTE  # noqa: E402

VOCAB = [
    "hello",                 # 0: ascii, never a Han ban candidate
    raw_surface(" 中"),       # 1: a Han character after a space marker
    raw_surface("한"),        # 2: Hangul stays (Korean text is what the ban protects)
    raw_surface("中가"),      # 3: Han + Hangul mixed - banned, the Han would still print
    raw_surface("あ中"),      # 4: Han + kana mixed - banned for the same reason
    "中",                    # 5: a non-encoded surface still checks out
    "Ā",                     # 6: one shifted byte (0x80), no Han
]


class Surfaces(unittest.TestCase):
    def test_byte_map_matches_the_server_tokenizer(self):
        self.assertEqual({v: k for k, v in BYTE_TO_UNICODE.items()}, ORACLE_UNICODE_TO_BYTE)

    def test_surface_round_trip(self):
        for text in ["hello", "Ġ中", "한글", "Ā", "\n", ""]:
            self.assertEqual(surface(raw_surface(text)), text)

    def test_contains_han(self):
        for text in ["中", "Ġ中", "中가", "あ中", "㐀", "豈", "\U00020000"]:
            self.assertTrue(contains_han(text), text)
        for text in ["", "plain", "한", "가나다", "あ", "぀", "ヿ", "\U0001F600"]:
            self.assertFalse(contains_han(text), text)

    def test_han_token_ids(self):
        self.assertEqual(han_token_ids(VOCAB), [1, 3, 4, 5])


class Configure(unittest.TestCase):
    def service_with(self, tokens, defaults=None):
        return SimpleNamespace(tok=SimpleNamespace(tokens=tokens),
                               sampling_defaults={} if defaults is None else defaults)

    def test_off_leaves_the_defaults_untouched(self):
        service = self.service_with(VOCAB)
        configure_cjk_ban(service, False)
        self.assertEqual(service.sampling_defaults, {})
        self.assertEqual(service.cjk_ban_tokens, 0)

    def test_on_bans_every_han_token_as_default(self):
        service = self.service_with(VOCAB)
        configure_cjk_ban(service, True)
        self.assertEqual(service.sampling_defaults["logit_bias"], {1: -100.0, 3: -100.0, 4: -100.0, 5: -100.0})
        self.assertEqual(service.cjk_ban_tokens, 4)

    def test_a_config_bias_stays_authoritative(self):
        service = self.service_with(VOCAB, defaults={"logit_bias": {2: 1.5}})
        configure_cjk_ban(service, True)
        self.assertEqual(service.sampling_defaults["logit_bias"], {2: 1.5})
        self.assertEqual(service.cjk_ban_tokens, 0)

    def test_bad_config_values_fail_closed(self):
        for value in ["true", None, 1]:
            with self.subTest(value=repr(value)), self.assertRaises(SystemExit):
                configure_cjk_ban(self.service_with(VOCAB), value)
        with self.assertRaises(SystemExit):          # nothing to ban: a tokenizer without Han is a config error
            configure_cjk_ban(self.service_with([raw_surface("only"), raw_surface("latin")]), True)
        with self.assertRaises(SystemExit):          # on without a tokenizer: say so, never half a ban
            configure_cjk_ban(SimpleNamespace(tok=None, sampling_defaults={}), True)


class Installed(unittest.TestCase):
    def test_the_server_reads_the_config_key(self):
        source = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        self.assertIn('from serve.cjk_ban import configure_cjk_ban  # STRATA_CJK_BAN_V1', source)
        self.assertIn('configure_cjk_ban(svc, cfg.get("ban_cjk_tokens", False))  # STRATA_CJK_BAN_V1', source)


if __name__ == "__main__":
    unittest.main()
