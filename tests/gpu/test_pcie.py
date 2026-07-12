"""PCIe-bench unit tests -- run INSIDE the vLLM image (`make test-pcie`).

pcie_bw_bench.py imports torch, a GPU/container dependency that is intentionally
absent from the hermetic host venv (see tests/unit). These tests therefore live in
a separate tier: `make test-pcie` runs pytest inside the vLLM image, where torch +
the GPU are present and the bench fails loudly when no GPU is detected. The helpers
under test (fmt_gbs, bw_at) use only math; the module import is what forces the
container tier, not the test logic."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pcie = _load_module("scripts/pcie_bw_bench.py", "pcie_bw_bench")


# --------------------------------------------------------------------------- #
# pcie_bw_bench.fmt_gbs + bw_at: pure formatter + closest-payload picker
# --------------------------------------------------------------------------- #
def test_fmt_gbs():
    assert pcie.fmt_gbs(1e9) == "1.00 GB/s"
    assert pcie.fmt_gbs(1.5e9) == "1.50 GB/s"


def test_bw_at_picks_closest_by_log_size():
    table = {1000: 1.0, 1_000_000: 2.0}
    bw, size = pcie.bw_at(700_000, table)  # log(700k) ~ 13.5, closer to 1e6 (13.8) than 1e3 (6.9)
    assert size == 1_000_000
    assert bw == 2.0
