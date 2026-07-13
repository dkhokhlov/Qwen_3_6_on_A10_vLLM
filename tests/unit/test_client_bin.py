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


# --------------------------------------------------------------------------- #
# Context-window / max_tokens env: the wrapper must tell Claude Code the REAL
# upstream window and cap output, or compaction is calibrated to a 200k default and
# max_tokens defaults to 32000 -> both overflow the 128k/64k upstream (the 400 bug).
# --------------------------------------------------------------------------- #
def _stack_envs(script: str) -> dict:
    """model -> {CTX_WINDOW, MAX_OUTPUT} from the wrapper's per-stack case block."""
    envs = {}
    for line in script.splitlines():
        s = line.strip()
        for stack in ("moe", "dense"):
            if s.startswith(f"{stack})") and "BASE=" in s:
                ctx = int(s.split("CTX_WINDOW=")[1].split(";")[0])
                out = int(s.split("MAX_OUTPUT=")[1].split(" ;;")[0].split()[0])
                envs[stack] = {"ctx": ctx, "out": out}
    return envs


def test_claude_qwen_exports_compaction_window_and_output_cap():
    for needle in ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_CODE_MAX_OUTPUT_TOKENS"):
        assert needle in SCRIPT, f"bin/claude-qwen must export {needle} (400-root-cause fix)"


def test_claude_qwen_window_matches_vllm_max_model_len():
    # CLAUDE_CODE_AUTO_COMPACT_WINDOW per stack must equal the compose --max-model-len, or
    # compaction triggers against the wrong window (the 400 root cause).
    import yaml
    envs = _stack_envs(SCRIPT)
    for stack, compose in (("moe", "docker-compose.moe.yaml"), ("dense", "docker-compose.yaml")):
        toks = yaml.safe_load((REPO / compose).read_text())["services"]["vllm"]["command"].split()
        window = int(toks[toks.index("--max-model-len") + 1])
        assert envs[stack]["ctx"] == window, f"{stack}: wrapper window {envs[stack]['ctx']} != --max-model-len {window}"


def test_claude_qwen_output_cap_fits_compaction_constraint():
    # PCT=80 (settings.json): cap must satisfy PCT*WINDOW + cap <= WINDOW -> cap <= WINDOW/5.
    envs = _stack_envs(SCRIPT)
    for stack, env in envs.items():
        assert env["out"] <= env["ctx"] // 5, (
            f"{stack}: output cap {env['out']} must be <= WINDOW/5 ({env['ctx'] // 5}) for PCT=80")
        assert env["out"] > 0
