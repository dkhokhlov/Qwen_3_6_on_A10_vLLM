"""Hermetic checks that the client-side DuckDuckGo search workaround is fully gone.

Server-side websearch_interception (LiteLLM + SearXNG) replaced the old per-session
DDG MCP server + `--disallowedTools WebSearch` block in bin/claude-qwen. These guard
against that dead client path creeping back (and its two obsolete files returning)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (REPO / "bin" / "claude-qwen").read_text()


def test_claude_qwen_has_no_client_side_search_block():
    # The wrapper must NOT re-introduce the DDG MCP / WebSearch-disabling client path.
    for needle in ("ddg-search", "qwen-mcp.json", "--disallowedTools WebSearch",
                   "CLAUDE_EXTRA_ALLOWED", "QWEN_MCP_CONFIG"):
        assert needle not in SCRIPT, (
            f"bin/claude-qwen still references {needle!r} -- server-side "
            "websearch_interception replaced the client-side DDG workaround")
    # ... and it must exec plain claude (server-side search needs no client wiring).
    assert "exec claude" in SCRIPT


def test_dead_ddg_files_are_gone():
    for name in ("ddg_search_mcp.py", "qwen-mcp.json"):
        assert not (REPO / "bin" / name).exists(), (
            f"bin/{name} is dead under server-side search -- remove it (and its ~/bin copy)")
