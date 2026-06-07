"""Check that PyPI and conda-forge jax dependency pins are in sync."""

import re
import urllib.error
import urllib.request
from pathlib import Path

import tomllib
import yaml
from packaging.version import Version

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def warn(msg: str) -> None:
    print(f"::warning ::{msg}")


def strip_jinja(text: str) -> str:
    return re.sub(r"{%[^%]*%}", "", text)


def extract_pin(dep: str) -> tuple[str, str] | None:
    if m := re.match(r"jax\s*(>=|==|<=|!=|~=)\s*([\w.]+)", dep):
        return m.group(1), m.group(2)
    return None


def main() -> None:
    # 1. Parse pyproject.toml
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    pypi_pin = None
    for dep in pyproject.get("project", {}).get("dependencies", []):
        if pin := extract_pin(dep):
            pypi_pin = pin
            break

    if pypi_pin is None:
        warn("could not find jax dependency in pyproject.toml")
        return

    # 2. Fetch and parse conda-forge recipe
    url = (
        "https://raw.githubusercontent.com/"
        "conda-forge/optax-feedstock/main/recipe/meta.yaml"
    )
    try:
        req = urllib.request.urlopen(url, timeout=10)
        raw = req.read().decode()
    except urllib.error.URLError as e:
        warn(f"unable to fetch conda-forge recipe ({e}); skipping sync check")
        return

    meta = yaml.safe_load(strip_jinja(raw))
    conda_pin = None
    for dep in (meta.get("requirements", {}) or {}).get("run", []):
        if pin := extract_pin(dep):
            conda_pin = pin
            break

    if conda_pin is None:
        warn("could not find jax pin in conda-forge recipe")
        return

    # 3. Compare operator and version
    (op_pypi, ver_pypi), (op_conda, ver_conda) = pypi_pin, conda_pin
    if op_pypi != op_conda or Version(ver_pypi) != Version(ver_conda):
        warn(f"jax pin mismatch - PyPI: jax{op_pypi}{ver_pypi}, "
             f"conda: jax{op_conda}{ver_conda}")
        print(
            "Please open a PR at "
            "https://github.com/conda-forge/optax-feedstock "
            f"to update recipe/meta.yaml: jax "
            f"{op_pypi}{ver_pypi}, jaxlib {op_pypi}{ver_pypi}"
        )
        return

    print(f"conda-forge jax pin matches PyPI (jax{op_pypi}{ver_pypi}).")


if __name__ == "__main__":
    main()
