"""Pure-Python helpers of the re-execution (no pandas, so the host can import them)."""

from __future__ import annotations

ENVIRONMENT_HINTS = ("ModuleNotFoundError", "ImportError", "No module named", "cannot import name", "pkg_resources",
                     "Could not find a version", "DLL load failed", "GLIBC", "libstdc++", "version `GLIBC")


def guess_failure_cause(log_text: str) -> str:
    """A first guess at environment against the agent's own logic, from the error text; the repair note gives the real answer."""
    return "environment" if any(hint in log_text for hint in ENVIRONMENT_HINTS) else "logic"


def package_differences(recorded: list[str], installed: list[str]) -> dict:
    """Packages the run recorded against those in the image used for the re-execution (`name==version` lines)."""
    def parse(lines):
        out = {}
        for line in lines:
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                out[name.strip().lower().replace("_", "-")] = version.strip()
        return out
    a, b = parse(recorded), parse(installed)
    for ignored in ("pip", "setuptools", "wheel"):
        a.pop(ignored, None), b.pop(ignored, None)
    return {"missing_in_image": sorted(set(a) - set(b)), "extra_in_image": sorted(set(b) - set(a)),
            "version_differs": sorted(f"{k}: run {a[k]}, image {b[k]}" for k in set(a) & set(b) if a[k] != b[k])}
