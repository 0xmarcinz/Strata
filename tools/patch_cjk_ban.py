"""Install the opt-in Han default ban into a server.py this repository does not own (the NVFP4 fork on Windows).

The fork is downloaded and kept as it is, so its `serve/server.py` is patched in place at launch, the way the
other fork scripts do it: two lines are added at anchors that both this tree and the fork carry, and
`serve/cjk_ban.py` is copied beside it. The patch is idempotent (a second run changes nothing), refuses a
source it does not recognise instead of guessing, and writes every file through a temporary name so a failure
leaves the fork exactly as it was.

`serve/logit_bias.py` is NOT replaced: a fork may carry its own. The script only checks that the fork's copy
remembers a large map between requests (the fork's own `_last`/`_CACHE_MIN`, or an equivalent), because a
55,000-entry ban normalized five times per request costs about 100 ms each time.

Usage (from the Windows launcher, before the fork's server starts):

    python tools/patch_cjk_ban.py %FORK_ROOT%\\serve\\server.py
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MARKER = "# STRATA_CJK_BAN_V1"
IMPORT_ANCHOR = "from serve import responses as responses_api  # noqa: E402"
CONFIG_ANCHOR = ('    recovery = cfg.get("reasoning_loop_recovery", False)   '
                 '# #728: false (default) | "stop" | "recover" (true)')
IMPORT = "from serve.cjk_ban import configure_cjk_ban  " + MARKER
CONFIG = '    configure_cjk_ban(svc, cfg.get("ban_cjk_tokens", False))  ' + MARKER
# The whole anchor line is re-emitted, so a second patch script that anchors on the neighbouring line keeps working.
CHANGES = (
    (IMPORT_ANCHOR, IMPORT_ANCHOR + "\n" + IMPORT),
    (CONFIG_ANCHOR, CONFIG_ANCHOR + "\n" + CONFIG),
)


class PatchError(ValueError):
    """The target does not look like a server.py this patch knows how to change."""


def patch_server(source: str) -> str:
    """`source` with the two hook lines in it; an already-patched source is returned unchanged."""
    if MARKER in source:
        if source.count(MARKER) != 2 or not all(part in source for part in (IMPORT, CONFIG)):
            raise PatchError("partial CJK-ban hook; inspect the target before editing it again")
        return source
    counts = [source.count(old) for old, _ in CHANGES]
    if counts != [1, 1]:
        raise PatchError(f"unknown server layout: expected one of each anchor, found {counts}")
    for old, new in CHANGES:
        source = source.replace(old, new, 1)
    return source


def _write_atomic(path: Path, text: str, newline: str, bom: str) -> None:
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as out:
            out.write(bom + text.replace("\n", newline))
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def install(server, module=None) -> dict:
    """Patch `server`, copy `module` beside it, and say what changed. Nothing is written when anything fails."""
    server = Path(server).resolve()
    module = Path(module or ROOT / "serve" / "cjk_ban.py").resolve()
    if not server.is_file():
        raise PatchError(f"required source is missing: {server}")
    if not module.is_file():
        raise PatchError(f"required source is missing: {module}")
    original = server.read_bytes()
    text = original.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    patched = patch_server(text.replace("\r\n", "\n"))
    compile(patched, str(server), "exec")                      # a broken patch never reaches the fork
    payload = module.read_text(encoding="utf-8-sig")
    compile(payload, module.name, "exec")
    destination = server.with_name(module.name)
    changed = 0
    if patched != text.replace("\r\n", "\n"):
        _write_atomic(server, patched, newline, "\ufeff" if original.startswith(b"\xef\xbb\xbf") else "")
        changed += 1
    if not destination.is_file() or destination.read_bytes() != module.read_bytes():
        _write_atomic(destination, payload, newline, "")
        changed += 1
    return {"changed": changed, "server": str(server), "module": str(destination),
            "bias_memo": _bias_memo(server)}


def _bias_memo(server: Path) -> str:
    """Whether the target's logit_bias.py remembers a large map: "yes", "no" or "absent" (a warning, never a write)."""
    sibling = server.with_name("logit_bias.py")
    if not sibling.is_file():
        return "absent"
    text = sibling.read_text(encoding="utf-8", errors="replace")
    return "yes" if ("_CACHE_MIN" in text or "_last" in text or "lru_cache" in text) else "no"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path, help="the fork's serve/server.py")
    parser.add_argument("--module", type=Path, help="serve/cjk_ban.py to copy (default: this tree's)")
    args = parser.parse_args(argv)
    try:
        result = install(args.server, args.module)
    except (OSError, UnicodeError, ValueError, SyntaxError) as e:
        print("[strata] cannot install the CJK token ban: " + str(e), file=sys.stderr)
        return 2
    print("[strata] CJK token ban ready; changed files=" + str(result["changed"]))
    if result["bias_memo"] == "no":
        print("[strata] note: this fork's serve/logit_bias.py does not remember a large map between requests, "
              "so a 55,000-entry ban is normalized on every request (about 100 ms). Update it from this tree "
              "if that matters here.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
