# Release Process

This document describes the steps required to publish a new optax release.

## Before the release

1.  Ensure all new features and bug fixes have been merged to `main`.
2.  Verify that the [CI tests](https://github.com/google-deepmind/optax/actions) are passing on `main`.
3.  Decide on the new version number following [semantic versioning](https://semver.org/).

## Step-by-step release

### 1. Update dependencies in `pyproject.toml`

If the minimum version of any dependency (e.g. `jax`, `jaxlib`) has changed, update the pins in:

- `pyproject.toml` — the canonical source of truth for PyPI.
- `.github/workflows/tests.yml` — the CI matrix uses a matching minimum version.

The comment in `pyproject.toml` lists all locations that must stay in sync.

### 2. Create a GitHub release

1.  Go to the [Releases page](https://github.com/google-deepmind/optax/releases) and click **Draft a new release**.
2.  Set the tag to `v<version>` (e.g. `v0.2.9`).
3.  Generate release notes from the recent commits.
4.  Publish the release.

The `pypi-publish.yml` workflow will automatically build and upload the package to PyPI.

### 3. Verify the PyPI publication

```bash
pip install optax==<version>
```

Check that the correct version is installed and that the dependencies resolve correctly.

### 4. Sync conda-forge feedstock

The [conda-forge/optax-feedstock](https://github.com/conda-forge/optax-feedstock) repository mirrors optax for conda users.

- A bot usually auto-updates the `version` and `sha256` fields within hours of a PyPI release.
- **However**, if the **dependency bounds** changed (e.g. `jax>=0.4.27` → `jax>=0.5.3`), the bot will **not** update them. You must open a manual PR against `conda-forge/optax-feedstock` modifying `recipe/meta.yaml`.

To verify the conda recipe is in sync:

```bash
# Compare jax/jaxlib pins
grep 'jax ' pyproject.toml
grep 'jax ' <(curl -s https://raw.githubusercontent.com/conda-forge/optax-feedstock/main/recipe/meta.yaml)
```

If they differ, open a PR on the feedstock repo matching the `pyproject.toml` pins exactly.

### 5. Verify the conda publication

```bash
conda install optax=<version> -c conda-forge
```

### 6. Update the documentation (if needed)

If the release adds new public APIs or changes existing ones, ensure the docs are up to date. The documentation is built automatically by ReadTheDocs.

## Checklist template

Copy this into the release issue or PR description:

```
- [ ] Release tagged on GitHub
- [ ] PyPI package published and verified
- [ ] Conda-forge recipe dependency bounds match pyproject.toml
- [ ] Conda package installable and working
- [ ] Documentation up to date
```
