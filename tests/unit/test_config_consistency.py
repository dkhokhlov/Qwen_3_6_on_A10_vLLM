"""Config-consistency tests: parse the 4 YAML files and assert cross-file invariants
that, when they drift, cause real outages (wrong model name, stale image, unreachable
host, missing flavor). Parametrized over both stacks so each gets its own test id."""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

# (name, compose path, litellm path, base model name)
STACKS = [
    pytest.param(REPO / "docker-compose.yaml", REPO / "litellm_config.yaml", "qwen3.6-27b", id="dense"),
    pytest.param(REPO / "docker-compose.moe.yaml", REPO / "litellm_config.moe.yaml", "qwen3.6-35b-a3b", id="moe"),
]


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _cmd_tokens(compose: dict) -> list:
    return compose["services"]["vllm"]["command"].split()


def _flag(tokens: list, flag: str) -> str:
    return tokens[tokens.index(flag) + 1]


# --------------------------------------------------------------------------- #
# Per-stack invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_served_model_name_equals_bare_litellm_model(compose_p, litellm_p, base):
    compose, litellm = _load(compose_p), _load(litellm_p)
    served = _flag(_cmd_tokens(compose), "--served-model-name")
    names = [m["model_name"] for m in litellm["model_list"]]
    assert served == base
    assert served in names, f"--served-model-name {served!r} not in litellm model_list {names}"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_three_flavors_present(compose_p, litellm_p, base):
    litellm = _load(litellm_p)
    names = {m["model_name"] for m in litellm["model_list"]}
    assert names == {base, f"{base}-preserve", f"{base}-nothink"}


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_model_info_max_output_tokens_matches_server_clamp(compose_p, litellm_p, base):
    # Claude Code's gateway discovery ignores model_info.max_output_tokens (it reads only
    # id/display_name from /v1/models), so the REAL cap is the server-side clamp env on
    # the litellm service. model_info is just /v1/models metadata for other clients; it
    # must at least agree with the clamp so the advertised limit isn't a lie.
    compose, litellm = _load(compose_p), _load(litellm_p)
    cap = int(compose["services"]["litellm"]["environment"]["CLAUDE_QWEN_MAX_TOKENS_CAP"])
    for m in litellm["model_list"]:
        assert m["model_info"]["max_output_tokens"] == cap, m["model_name"]


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_inject_streamed_usage_env_present_and_on(compose_p, litellm_p, base):
    # Auto-compaction never fires unless the proxy reports accurate streamed
    # message_start.usage.input_tokens (Claude Code's tracker reads the STREAMED start
    # usage, not the terminal result). CLAUDE_QWEN_INJECT_STREAMED_USAGE gates the
    # preflight /tokenize + inject fix; it must be present and "1" in both stacks, or
    # compaction silently never fires (the original unbounded-growth-to-400 regression).
    env = _load(compose_p)["services"]["litellm"]["environment"]
    assert env.get("CLAUDE_QWEN_INJECT_STREAMED_USAGE") == "1", (
        "CLAUDE_QWEN_INJECT_STREAMED_USAGE must be \"1\" so auto-compact fires"
    )


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_max_tokens_cap_fits_under_compaction_window(compose_p, litellm_p, base):
    # The 400 root cause: Claude Code assumes a 200k window and sizes compaction to it,
    # so for a 128k/64k upstream compaction fires PAST the wall and the request overflows.
    # CLAUDE_CODE_AUTO_COMPACT_WINDOW (set in bin/claude-qwen) re-targets compaction to the
    # real window; the cap must then satisfy PCT*WINDOW + cap <= WINDOW so the compaction
    # request itself fits. PCT is 80 (settings.json CLAUDE_AUTOCOMPACT_PCT_OVERRIDE),
    # WINDOW is vLLM's --max-model-len, cap is the server clamp env -> cap <= WINDOW/5.
    toks = _cmd_tokens(_load(compose_p))
    window = int(_flag(toks, "--max-model-len"))
    cap = int(_load(compose_p)["services"]["litellm"]["environment"]["CLAUDE_QWEN_MAX_TOKENS_CAP"])
    assert cap <= window // 5, f"{base}: cap {cap} must be <= WINDOW/5 ({window // 5}) for PCT=80"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_litellm_params_use_env_indirection(compose_p, litellm_p, base):
    litellm = _load(litellm_p)
    for m in litellm["model_list"]:
        lp = m["litellm_params"]
        assert lp["model"] == "os.environ/LITELLM_VLLM_MODEL", m["model_name"]
        assert lp["api_base"] == "os.environ/LITELLM_VLLM_API_BASE", m["model_name"]
        assert lp["api_key"] == "dummy", m["model_name"]


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_container_name_matches_backend_container_env(compose_p, litellm_p, base):
    compose = _load(compose_p)
    vllm = compose["services"]["vllm"]
    litellm_env = compose["services"]["litellm"]["environment"]
    assert vllm["container_name"] == litellm_env["BACKEND_CONTAINER"]


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_litellm_api_base_points_at_vllm(compose_p, litellm_p, base):
    compose = _load(compose_p)
    toks = _cmd_tokens(compose)
    api_base = compose["services"]["litellm"]["environment"]["LITELLM_VLLM_API_BASE"]
    assert api_base == f"http://vllm:{_flag(toks, '--port')}/v1"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_openwebui_points_at_litellm(compose_p, litellm_p, base):
    compose = _load(compose_p)
    env = compose["services"]["open-webui"]["environment"]
    assert env["OPENAI_API_BASE_URL"] == "http://litellm:4000/v1"
    assert env["OPENAI_API_KEY"] == "anything"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_docker_api_network_membership(compose_p, litellm_p, base):
    compose = _load(compose_p)
    svc = compose["services"]
    # docker-api is internal-only; its members are the proxy + litellm (list-form networks)
    assert compose["networks"]["docker-api"]["internal"] is True
    assert "docker-api" in svc["docker-sock-proxy"]["networks"]
    assert "docker-api" in svc["litellm"]["networks"]
    # vllm + open-webui stay on the default network (no explicit networks block)
    assert "networks" not in svc["vllm"]
    assert "networks" not in svc["open-webui"]


# --------------------------------------------------------------------------- #
# Silent-failure / outage regressions (each maps to a real fix in git history).
# These are the invariants whose drift causes no test failure and no error --
# just a broken stack (unhealthy container, un-waked vLLM, OOM staying down,
# a root-equivalent socket on the wrong service). All parse YAML; all hermetic.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_vllm_parsers_are_qwen3_xml_and_qwen3(compose_p, litellm_p, base):
    # --tool-call-parser hermes leaked tool calls as raw text; qwen3_xml is correct.
    toks = _cmd_tokens(_load(compose_p))
    assert _flag(toks, "--tool-call-parser") == "qwen3_xml"
    assert _flag(toks, "--reasoning-parser") == "qwen3"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_healthchecks_never_invoke_bare_python(compose_p, litellm_p, base):
    # Bare `python` exits 127 on vLLM images (python3 only) -> permanent unhealthy.
    for name, svc in _load(compose_p)["services"].items():
        cmd = svc.get("healthcheck", {}).get("test", [""])[-1]
        assert " python " not in f" {cmd} ", f"{name} healthcheck must not invoke bare python"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_vllm_restart_is_unless_stopped(compose_p, litellm_p, base):
    # MoE once shipped restart:"no" -> an OOM crash stayed down until manual restart.
    assert _load(compose_p)["services"]["vllm"]["restart"] == "unless-stopped"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_startup_hook_uses_colon_form(compose_p, litellm_p, base):
    # A `module.function` (dot) spec silently never resolves; loader wants `module:function`.
    val = _load(compose_p)["services"]["litellm"]["environment"]["LITELLM_WORKER_STARTUP_HOOKS"]
    assert ":" in val and "." not in val, f"startup hook must be module:function colon form; got {val!r}"
    assert val.endswith(":start_background_tasks")


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_proxy_healthchecks_probe_correct_endpoints(compose_p, litellm_p, base):
    svc = _load(compose_p)["services"]
    litellm_hc = svc["litellm"]["healthcheck"]["test"][-1]
    openwebui_hc = svc["open-webui"]["healthcheck"]["test"][-1]
    assert "4000/health/liveliness" in litellm_hc   # /health would NOT report liveness
    assert "8080/health" in openwebui_hc


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_vllm_and_litellm_publish_on_all_interfaces(compose_p, litellm_p, base):
    svc = _load(compose_p)["services"]
    assert "0.0.0.0:8000:8000" in svc["vllm"]["ports"]   # bench reads /metrics; remote hosts hit :4000
    assert "0.0.0.0:4000:4000" in svc["litellm"]["ports"]


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_litellm_settings_drops_params_and_runs_one_worker(compose_p, litellm_p, base):
    s = _load(litellm_p)["litellm_settings"]
    assert s["drop_params"] is True   # drop Claude Code params vLLM rejects -> no 400s
    assert s["num_workers"] == 1      # one idle-watcher, one wake-lock owner


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_litellm_settings_callbacks_are_builtin_names(compose_p, litellm_p, base):
    # litellm_settings.callbacks:<module.handler> (a DOTTED string) is a SILENT NO-OP:
    # the resolver only matches BUILT-IN integration names (e.g. "websearch_interception").
    # The custom wake/idle handler must come from the startup hook, never a dotted string.
    s = _load(litellm_p)["litellm_settings"]
    for c in s.get("callbacks", []):
        assert "." not in c, f"callbacks entry {c!r} looks like a module.handler string (silent no-op)"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_raw_docker_socket_mounted_only_on_sock_proxy(compose_p, litellm_p, base):
    # The socket is root-equivalent; only docker-sock-proxy may hold it. litellm (and
    # every other service) must route container control over HTTP via the sidecar.
    services = _load(compose_p)["services"]
    assert any("docker.sock" in str(v) for v in services["docker-sock-proxy"].get("volumes", []))
    for name, svc in services.items():
        if name == "docker-sock-proxy":
            continue
        mounts = [str(v) for v in svc.get("volumes", [])]
        assert not any("docker.sock" in m for m in mounts), \
            f"{name} must not mount the raw docker socket (route via docker-sock-proxy)"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_sock_proxy_completeness(compose_p, litellm_p, base):
    env = _load(compose_p)["services"]["docker-sock-proxy"]["environment"]
    assert env["VERSION"] == "0"   # revoke the info-leaking upstream default
    assert env["PING"] == "1"      # backs the healthcheck
    assert "@sha256:" in _load(compose_p)["services"]["docker-sock-proxy"]["image"]


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_litellm_routes_container_control_through_sock_proxy(compose_p, litellm_p, base):
    svc = _load(compose_p)["services"]["litellm"]
    assert svc["environment"]["DOCKER_API_BASE"] == "http://docker-sock-proxy:2375"
    assert "docker-sock-proxy" in svc.get("depends_on", [])


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_websearch_interception_configured(compose_p, litellm_p, base):
    # Server-side web search needs the built-in callback + a searxng search_tool, gated to
    # hosted_vllm (Qwen only). Drift here -> WebSearch silently inert again under Qwen.
    litellm = _load(litellm_p)
    s = litellm["litellm_settings"]
    assert "websearch_interception" in s.get("callbacks", []), "websearch_interception missing"
    assert "hosted_vllm" in s["websearch_interception_params"]["enabled_providers"]
    tools = litellm.get("search_tools", [])
    searxng = next((t for t in tools if t["litellm_params"].get("search_provider") == "searxng"), None)
    assert searxng is not None, "no searxng entry in search_tools"
    assert searxng["litellm_params"]["api_base"] == "os.environ/SEARXNG_API_BASE"


@pytest.mark.parametrize("compose_p, litellm_p, base", STACKS)
def test_searxng_service_wired_to_litellm(compose_p, litellm_p, base):
    # searxng must be a service litellm depends on + reaches, but NOT on the internal
    # no-egress docker-api net (it must reach DuckDuckGo). settings.yml mounted read-only.
    svc = _load(compose_p)["services"]
    assert "searxng" in svc
    assert "default" in svc["searxng"].get("networks", [])
    assert "docker-api" not in svc["searxng"].get("networks", []), \
        "searxng must not be on the internal no-egress docker-api network"
    assert svc["litellm"]["environment"]["SEARXNG_API_BASE"] == "http://searxng:8080"
    assert "searxng" in svc["litellm"].get("depends_on", [])
    mounts = [str(v) for v in svc["searxng"].get("volumes", [])]
    assert any("settings.yml" in m and ":ro" in m for m in mounts), mounts


# --------------------------------------------------------------------------- #
# Cross-stack invariants (must be identical across both compose files)
# --------------------------------------------------------------------------- #
def _composes():
    return _load(REPO / "docker-compose.yaml"), _load(REPO / "docker-compose.moe.yaml")


def test_vllm_image_pinned_identically_across_stacks():
    a, b = _composes()
    img_a = a["services"]["vllm"]["image"]
    img_b = b["services"]["vllm"]["image"]
    assert img_a == img_b
    assert "@sha256:" in img_a, "vLLM image must be pinned by digest"


def test_docker_sock_proxy_identical_across_stacks():
    a, b = _composes()
    sa, sb = a["services"]["docker-sock-proxy"], b["services"]["docker-sock-proxy"]
    assert sa["image"] == sb["image"]
    assert sa["environment"] == sb["environment"]


def test_docker_sock_proxy_allowlist_is_least_privilege():
    a, _ = _composes()
    env = a["services"]["docker-sock-proxy"]["environment"]
    assert env["POST"] == "1" and env["ALLOW_START"] == "1" and env["ALLOW_STOP"] == "1"
    assert env["CONTAINERS"] == "0"  # no list/inspect/create/delete
    assert env["EVENTS"] == "0"      # no info leak


def test_searxng_image_pinned_by_digest():
    # Match the repo convention (vllm + docker-sock-proxy): pin by digest so a deploy is
    # reproducible and a moved :latest tag can't silently change behavior.
    a, b = _composes()
    for name, compose in [("dense", a), ("moe", b)]:
        img = compose["services"]["searxng"]["image"]
        assert "@sha256:" in img, f"{name}: searxng image must be pinned by digest; got {img!r}"
