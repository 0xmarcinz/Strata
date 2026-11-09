# Per-request token biases

`POST /v1/chat/completions` accepts `logit_bias` with a supporting CUDA, HIP or Metal
engine, also with continuous batching (each slot keeps its own bias). The same bias applies to every generated
token in the request, including tokens checked together during speculative
decoding. It does not force single-token verification windows.

## Example: Hello becomes Hi

We use `Hello` because it is a familiar word, it is exactly one token in the
tested tokenizer, and the unbiased model naturally chooses it for this prompt.
The tokenizer maps `Hello` to **token ID 9419**. That number identifies the token;
it is not its logit or probability. The value **-100** is the requested bias,
which this implementation treats as a hard exclusion.

Before: send this request without a bias. The measured response was **Hello**.

```json
{
  "model": "bias-validation",
  "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}],
  "max_tokens": 8,
  "temperature": 0,
  "reasoning_effort": "none"
}
```

After: add only `logit_bias`. The measured response was **Hi** (token 12675).

```json
{
  "model": "bias-validation",
  "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}],
  "max_tokens": 8,
  "temperature": 0,
  "reasoning_effort": "none",
  "logit_bias": {"9419": -100}
}
```

The equivalent llama.cpp-style pair list is `"logit_bias": [[9419, false]]`;
it also produced **Hi**. Removing the field on the next request restored **Hello**.
Nothing is banned by default. This is a per-request example, not a built-in list.
The `model` value above is the test server's alias; use your server's model name.
Numeric pair values are also supported, for example `[[42, 2.5]]`.
Token IDs belong to the loaded model's tokenizer; the example ID is not portable
between arbitrary models. Leading spaces and letter case can also change token IDs.
The four English requests were measured on ISTA IQ3_XXS; a
[reproduction script](../tools/logit_bias_english_example.py) is included.

## Semantics and limits

Biases must be finite numbers in `[-100, 100]`. Negative values discourage a
token, positive values encourage it, and **-100 excludes it** (`false` in a
pair list has the same meaning). The bias is added before repetition/frequency/
presence penalties, temperature, and candidate filtering. Raw model logits
are not overwritten. Banning a token does not ban a word or language: other
token sequences may produce the same text.

Omitting the field, `null`, `{}`, or `[]` clears the previous request's bias.
Biases are request sampling state; they are not stored in a conversation cache.
They do not change the model weights or prompt KV. A fixed dense vocabulary
vector is uploaded when the bias changes and reused within the request.

Malformed entries, duplicate IDs in pair lists, IDs outside the vocabulary,
out-of-range biases, or banning the entire vocabulary return HTTP 400. Validation
on the chat route occurs before SSE streaming headers are sent. Nonempty biases
also return HTTP 400 with older engines or the current SYCL engine, rather than
being silently ignored. The engine advertises
`logit_bias=1` in its `INFO` response when this implementation is present.

MTP target verification applies the bias; the draft model is not biased, so a
restrictive list may reduce draft acceptance. Ordinary MTP was tested. The
experimental coupled/rejection-sampling combinations and multi-GPU execution
were not validated in this campaign. There is no startup bias-file option in
this change: send the field with each request.

## A default ban for every request: `ban_cjk_tokens`

A deployment whose answers never contain Han characters (a Korean or Japanese assistant, for example) can
ban every token whose surface holds one, instead of asking each client to send the list. `serve/cjk_ban.py`
builds the map once at server startup from the server's own tokenizer, and the run config turns it on:

```json
{ "ban_cjk_tokens": true }
```

The map rides the run config's sampling defaults, which every route merges under a request's own fields, so
a request can still send its own `logit_bias` and an explicit `{}` clears the ban for that turn. A config
that already sets a sampling `logit_bias` keeps it and the ban is not installed; the startup line says so.
Hangul, kana and Latin are left alone; only Han ideographs (Extension A, the Unified block, the
compatibility block, and Extension B..F) are banned.

The ban is complete for token surfaces - a token can only emit what its surface holds - while a Han
character spelled out of byte-fallback tokens, which the model does for these characters about never, stays
possible. It is also model-wide rather than language-aware: a profile that must answer in Chinese cannot
use it as it stands.

Two costs, both once per request rather than once per token: the engine's command line carries the bias
string (about 660 KB for a 248,320-token Qwen vocabulary, where 55,328 tokens are Han), and the map has to
normalize. `serve/logit_bias.py` answers the same map object from its cache after the first call, so a
second request pays a copy, not a scan. Decode speed with the ban active has not been measured here; the MTP
note above applies to the ban list as well (the draft model is not biased).

### The NVFP4 fork on Windows

That fork is kept as it is, so `tools/patch_cjk_ban.py` adds the two hook lines to its `serve/server.py` at
launch and copies `serve/cjk_ban.py` beside it, the way the other fork scripts work. It never replaces the
fork's `serve/logit_bias.py`; it reports when that copy does not remember a large map between requests,
because then the 55,000-entry ban is normalized on every call (about 100 ms).

```bat
"%PYTHON%" "%STRATA_ROOT%\tools\patch_cjk_ban.py" "%FORK_ROOT%\serve\server.py"
if errorlevel 1 (echo Could not prepare the CJK token ban.& exit /b 32)
```
