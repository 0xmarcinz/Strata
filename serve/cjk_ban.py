"""Opt-in default ban of every token whose surface holds a Han character (config "ban_cjk_tokens").

A Qwen reply to a Korean or Japanese prompt sometimes mixes Han characters into the answer. The engine can
only exclude whole tokens (`logit_bias` -100), so this builds the ban once at startup from the server's own
tokenizer: a token is banned when its decoded surface holds a Han character. That is complete for tokens - a
token can only emit what its surface holds - while a Han character spelled out of byte-fallback tokens, which
the model does for these characters about never, stays possible.

The map rides the run config's sampling defaults, so every route shares it (serve/server.py merges
`sampling_defaults` under a request's own fields), and one request can still override it with its own
`logit_bias`: an explicit `{}` clears the ban for that turn, and a profile that already sets a `logit_bias`
keeps it, because that bias is the person's own choice.

The ban is model-wide, not language-aware: a deployment that must answer in Chinese on the same profile needs
its own bias list. Nothing is downloaded and nothing is measured: the scan is local and one-shot.

Vocab surfaces arrive in the byte-level BPE form (the GPT-2 byte encoder), so `surface` maps a stored token
back to text and passes a non-encoded token through unchanged.
"""
from __future__ import annotations

import time


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.

    The same map `tools/strata_tokenizer.py` stores (kept in sync by `serve/test_cjk_ban.py`): a stored vocab is
    only readable through that exact shift order.
    """
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

#: Han ideographs: Extension A, the Unified block, the compatibility block, and the Extension B..F planes
#: (where the surrogate pairs lived in UTF-16 storage).
_HAN = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F))


def surface(token: str) -> str:
    """The text a stored vocab entry stands for: the byte-encoder form decoded, a plain token unchanged."""
    try:
        return bytes(UNICODE_TO_BYTE[c] for c in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return token


def raw_surface(text: str) -> str:
    """`surface` reversed: how a tokenizer file stores this piece of text."""
    return "".join(BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))


def contains_han(text: str) -> bool:
    """True when `text` holds a Han character (Hangul, kana and Latin are left alone)."""
    for ch in text:
        o = ord(ch)
        if o > 0x7F and any(lo <= o <= hi for lo, hi in _HAN):
            return True
    return False


def han_token_ids(tokens) -> list:
    """Ids of the stored vocab whose decoded surface holds a Han character."""
    return [i for i, raw in enumerate(tokens)
            if not raw.isascii() and contains_han(surface(raw))]


def configure_cjk_ban(service, value):
    """Put the default Han ban on the run config's sampling defaults (off unless true).

    A profile or the person can still set `logit_bias` themselves: that block wins and this leaves it alone.
    """
    if not isinstance(value, bool):
        raise SystemExit('[strata] config "ban_cjk_tokens" must be true or false')
    service.cjk_ban_tokens = 0
    if not value:
        return
    tokens = getattr(getattr(service, "tok", None), "tokens", None)
    if not tokens:
        raise SystemExit('[strata] config "ban_cjk_tokens" needs the loaded tokenizer; '
                         "start the server with the model's own tokenizer")
    started = time.perf_counter()
    banned = {i: -100.0 for i in han_token_ids(tokens)}
    if not banned:
        raise SystemExit('[strata] config "ban_cjk_tokens": this tokenizer holds no Han token to ban')
    defaults = service.sampling_defaults
    if defaults.get("logit_bias"):
        print("[strata] ban_cjk_tokens: the config already sets a sampling logit_bias; it stays as it is",
              flush=True)
        return
    defaults["logit_bias"] = banned
    service.cjk_ban_tokens = len(banned)
    print(f"[strata] CJK ban: {len(banned)} of {len(tokens)} tokens excluded for every request "
          f"(ban_cjk_tokens; a request's own logit_bias overrides it), scanned in "
          f"{time.perf_counter() - started:.1f}s", flush=True)
