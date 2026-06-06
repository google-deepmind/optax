"""Check that PyPI and conda-forge jax dependency pins are in sync."""

import re
import sys
import urllib.request

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from packaging.version import Version


def warn(msg: str) -> None:
    print(f"::warning ::{msg}")


def main() -> None:
    # 1. Parse pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    jax_pin = None
    for dep in pyproject.get("project", {}).get("dependencies", []):
        m = re.match(r"jax\s*(>=|==|)\s*([\w.]+)", dep)
        if m:
            jax_pin = m.group(2)
            break

    if jax_pin is None:
        warn("could not find jax dependency in pyproject.toml")
        sys.exit(0)

    # 2. Fetch conda-forge recipe
    try:
        url = (
            "https://raw.githubusercontent.com/"
            "conda-forge/optax-feedstock/main/recipe/meta.yaml"
        )
        req = urllib.request.urlopen(url, timeout=10)
        conda_meta = req.read().decode()
    except Exception as e:
        warn(f"unable to fetch conda-forge recipe ({e}); skipping sync check")
        sys.exit(0)

    m = re.search(r"^\s*-\s+jax\s*>=\s*([\w.]+)", conda_meta, re.MULTILINE)
    if not m:
        warn("could not find jax pin in conda-forge recipe")
        sys.exit(0)
    conda_jax = m.group(1)

    # 3. Compare
    if Version(jax_pin) != Version(conda_jax):
        warn(f"jax pin mismatch - PyPI: {jax_pin}, conda: {conda_jax}")
        print(
            "Please open a PR at "
            "https://github.com/conda-forge/optax-feedstock "
            f"to update recipe/meta.yaml: jax >={jax_pin}, "
            f"jaxlib >={jax_pin}"
        )
        sys.exit(0)

    print(f"conda-forge jax pin matches PyPI ({jax_pin}).")


if __name__ == "__main__":
    main()
