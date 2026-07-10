"""Integration regression: the Open WebUI empty-model-list bug.

Root cause: Open WebUI persists the OpenAI connection base URL in its sqlite DB
(config.openai.api_base_urls). That row OVERRIDES OPENAI_API_BASE_URL on every
restart, so a stale value (e.g. http://gate:8000/v1) yields ClientConnectorDNSError
and an empty model dropdown -- even when the env var is correct. `make clean` wipes
the volume so the env var re-seeds the row; this test asserts the runtime agrees
with the compose env. This is the check that would have caught the original bug.

Read-only (docker exec SELECT + docker logs). Skipped unless :4000 answers.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


def _compose_env_base() -> str:
    for name in ("docker-compose.moe.yaml", "docker-compose.yaml"):
        f = REPO / name
        if f.exists():
            return yaml.safe_load(f.read_text())["services"]["open-webui"]["environment"]["OPENAI_API_BASE_URL"]
    raise RuntimeError("no compose file found")


def _db_openai_base_urls() -> list:
    sql = (
        "import sqlite3; c=sqlite3.connect('/app/backend/data/webui.db'); "
        "r=c.execute(\"select value from config where key='openai.api_base_urls'\").fetchone(); "
        "print(r[0] if r else '[]')"
    )
    out = subprocess.check_output(["docker", "exec", "open-webui", "python3", "-c", sql], text=True)
    return json.loads(out.strip())


def _boot_time() -> str:
    return subprocess.check_output(
        ["docker", "inspect", "-f", "{{.State.StartedAt}}", "open-webui"], text=True).strip()


def test_db_openai_base_url_matches_compose_env():
    # After `make clean` the row is either re-seeded from the env var or empty (env used
    # in-memory) -- both are "fixed". The bug is a NON-env entry overriding the env var,
    # so assert every persisted entry equals the compose value (an empty list passes).
    expected = _compose_env_base().rstrip("/")
    actual = _db_openai_base_urls()
    bad = [a for a in actual if a.rstrip("/") != expected]
    assert not bad, (
        f"DB openai.api_base_urls has non-env entries {bad!r} (env={expected!r}).\n"
        "A stale row here overrides OPENAI_API_BASE_URL -> the empty-model-list bug. "
        "Fix: `make clean` (resets the volume) or update the row.")


def test_no_unresolvable_host_errors_since_boot():
    logs = subprocess.check_output(
        ["docker", "logs", "open-webui", "--since", _boot_time()],
        text=True, stderr=subprocess.STDOUT)
    hits = re.findall(r"ClientConnectorDNSError|Name or service not known|gate:\d+", logs)
    assert not hits, f"open-webui still hitting an unresolvable host this boot: {hits[:3]}"
