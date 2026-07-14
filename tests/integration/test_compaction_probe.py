"""Integration: opt-in proxy-side compaction (compact_20260112 polyfill) fires REPEATEDLY.

Live end-to-end probe -- SLOW (~10-15 min: grows a real `claude -p --resume` session,
each compaction turn runs a summarization sub-call on vLLM) and needs the stack up with
the opt-in env ON. Skipped by default even under `make test-integration` (env-gated), so
it never runs in CI or a casual integration pass.

Prerequisites (operator sets these up; the test does not flip them):
  1. `make start35` (MoE stack up, vLLM warm).
  2. CLAUDE_QWEN_PROXY_COMPACT=1 on the litellm service (edit docker-compose.moe.yaml,
     `make start35` to recreate) -- the async_pre_request_hook then injects
     context_management on /v1/messages.
  3. `claude` on PATH; vLLM :8000/metrics reachable from the host.

Run explicitly:
  RUN_LIVE_COMPACTION_PROBE=1 make test-integration
  # or target only this file:
  RUN_LIVE_COMPACTION_PROBE=1 python3 -m pytest tests/integration/test_compaction_probe.py -m integration -o addopts="" -s

What it asserts (the plan-B success criteria):
  * The polyfill fires REPEATEDLY (>= 2 compaction resets in one session) -- this is what
    claude's own auto-compact CANNOT do (its per-session breaker caps it at one).
  * Every compaction is TRANSPARENT to claude: no `compact` system event (compact=False
    throughout) -- the polyfill rewrites server-side, no claude compact_boundary.
  * No 400 / timeout / non-zero exit (context stays bounded under --max-model-len).
  * claude's tracker (the proxy's injected message_start usage) resets each cycle -- the
    per-turn `in` drops sharply at each compaction.
To ISOLATE the polyfill, WINDOW is set huge so claude's own T4 threshold (window - max_output
- 13k) is unreachable; only the polyfill fires. See probe_compaction.py for the mechanism.
"""
import os

import pytest

import probe_compaction as P

pytestmark = [pytest.mark.integration, pytest.mark.live_compaction_probe]

# Isolation config: claude's T4 threshold = WINDOW - 256 - 13000 -> unreachable, so ONLY
# the polyfill (CLAUDE_QWEN_PROXY_COMPACT_THRESHOLD, 90000) fires. BLOB=18000 crosses 90000
# every ~5 turns -> >= 2 compactions in 12 turns.
WINDOW = 1_000_000
BLOB_TOKENS = 18_000
TURNS = 12
MAX_MODEL_LEN = 128_000   # moe --max-model-len; the polyfill must keep `in` under this


@pytest.fixture(autouse=True)
def _skip_unless_explicitly_enabled():
    # conftest._require_live_stack already skips integration tests when :4000 is down;
    # this gate additionally skips the ~10-15 min probe unless explicitly opted into, so
    # `make test-integration` does NOT run it by default.
    #
    # CI hard-exclude: even if RUN_LIVE_COMPACTION_PROBE=1 is set AND :4000 is up (a
    # self-hosted runner could satisfy both), never launch the 10-15 min live probe in a
    # CI context. GitHub Actions / most CI systems set the CI env var; skip on it.
    if os.environ.get("CI"):
        pytest.skip("live compaction probe is CI-excluded (10-15 min, needs a live stack)")
    if not os.environ.get("RUN_LIVE_COMPACTION_PROBE"):
        pytest.skip("set RUN_LIVE_COMPACTION_PROBE=1 to run the live compaction probe (~10-15 min)")


def test_proxy_compact_fires_repeatedly_and_transparently():
    result = P.run_probe(WINDOW=WINDOW, PCT=80, TURNS=TURNS, BLOB_TOKENS=BLOB_TOKENS,
                         turn_timeout=240)
    turns = result["turns"]
    assert turns, "probe produced no turns"
    # No failures / timeouts / 400s -- every recorded turn must be ok.
    bad = [t for t in turns if not t["ok"]]
    assert not bad, f"turns failed: {[(t['turn'], t['stderr'][:120]) for t in bad]}"

    # The polyfill is transparent: NO claude compact_boundary at any turn. (If claude's T4
    # were firing, compact=True would appear; with WINDOW huge it must not.)
    compact_turns = [t["turn"] for t in turns if t["compact"]]
    assert not compact_turns, f"unexpected claude compact events at turns {compact_turns} (WINDOW={WINDOW} should disable T4)"

    # Context bounded: no turn's reported input reaches --max-model-len (no overflow).
    over = [t["turn"] for t in turns if isinstance(t["in"], int) and t["in"] >= MAX_MODEL_LEN]
    assert not over, f"turns hit/over the {MAX_MODEL_LEN} wall: {over}"

    # REPEATED compaction: count turns where `in` dropped sharply vs the prior turn (a
    # polyfill reset). A normal turn grows by ~BLOB_TOKENS; a reset drops by far more.
    resets = []
    for a, b in zip(turns, turns[1:]):
        ai, bi = a["in"], b["in"]
        if isinstance(ai, int) and isinstance(bi, int) and bi < ai - 30_000:
            resets.append(b["turn"])
    assert len(resets) >= 2, (
        f"expected >= 2 repeated polyfill compactions, saw {len(resets)} at turns {resets}; "
        f"turns={[(t['turn'], t['in']) for t in turns]}"
    )

    # --- M2 cross-checks: validate the result via three INDEPENDENT signals, so a
    # silent injection/wiring failure can't pass on claude's reported numbers alone. ---

    # (1) session_id constancy: the probe resumes ONE session; a mid-probe change means
    # the --resume chain broke, silently invalidating the per-turn growth model.
    sids = {t["session_id"] for t in turns if t.get("session_id")}
    assert len(sids) == 1, f"session_id changed mid-probe (resume chain broke): {sids}"

    # (2) cross-check delta: vLLM's cumulative prompt_tokens is monotonic, so every turn's
    # delta must be >= 0. A negative delta means a broken /metrics read or a vLLM restart
    # mid-probe -- either invalidates the probe's numbers. (delta=0 is fine: a transient
    # missed read, carried forward by metric()'s None handling.)
    neg = [t["turn"] for t in turns
           if isinstance(t["delta"], (int, float)) and t["delta"] < 0]
    assert not neg, (
        f"negative vLLM prompt_tokens delta at turns {neg} (broken /metrics read or vLLM restart)")

    # (3) litellm log evidence: the litellm container must log "proxy-compact: injected
    # context_management" -- proof the opt-in hook fired on the /v1/messages path, not just
    # that claude's reported numbers happened to drop. Skipped if docker isn't reachable
    # from the probe host (an environment limitation, not a probe failure).
    n_logs = P.litellm_injection_log_count()
    if n_logs is not None:
        assert n_logs >= 1, (
            "no 'proxy-compact: injected' lines in litellm logs -- the opt-in hook never "
            "fired (CLAUDE_QWEN_PROXY_COMPACT not 1 on the litellm service?)")