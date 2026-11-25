# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keeps the Markdown previews of the Jupyter notebooks in sync.

Notebook diffs are hard to review, so each notebook tracked by git has a
Markdown preview in `.github/notebook-reviews/` that mirrors its path, e.g.
`examples/lbfgs.ipynb` -> `.github/notebook-reviews/examples/lbfgs.md`.

This script runs as a pre-commit hook, locally and in CI. It
1. regenerates the preview of every tracked notebook,
2. deletes the previews of deleted or renamed notebooks,
3. locally, adds the updated previews to the commit being made, so a single
   `git commit` contains both the notebooks and their previews;
4. in CI (where the `CI` environment variable is set), only checks: it fails
   if any preview is out of date, so that notebook changes cannot be merged
   without their previews.
"""

import os
import pathlib
import subprocess
import sys

import jupytext

PREVIEW_DIR = pathlib.Path(".github/notebook-reviews")

# Keep what matters in a review: drop the notebook metadata header (kernel,
# jupytext version) and Colab cell metadata (ids, execution info), but keep
# cell tags, which change how cells are executed and rendered in the docs.
JUPYTEXT_OPTIONS = {
    "notebook_metadata_filter": "-all",
    "cell_metadata_filter": "tags,-all",
}


def _git(*args: str) -> list[str]:
  """Runs a git command and returns its output lines."""
  result = subprocess.run(
      ["git", *args], check=True, capture_output=True, text=True
  )
  return result.stdout.splitlines()


def _render_preview(notebook: pathlib.Path) -> str:
  """Returns the Markdown preview of a notebook."""
  nb = jupytext.read(notebook)
  nb.metadata.setdefault("jupytext", {}).update(JUPYTEXT_OPTIONS)
  text = jupytext.writes(nb, fmt="md")
  return text if text.endswith("\n") else text + "\n"


def main() -> int:
  os.chdir(_git("rev-parse", "--show-toplevel")[0])

  expected = set()
  for path in _git("ls-files", "--", "*.ipynb"):
    notebook = pathlib.Path(path)
    if not notebook.exists():  # Deleted, but the deletion is not staged.
      continue
    preview = PREVIEW_DIR / notebook.with_suffix(".md")
    expected.add(preview)
    text = _render_preview(notebook)
    if not preview.exists() or preview.read_text(encoding="utf-8") != text:
      preview.parent.mkdir(parents=True, exist_ok=True)
      with open(preview, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

  for preview in PREVIEW_DIR.rglob("*.md"):
    if preview not in expected:
      preview.unlink()

  # Compare against the git index: previews that were changed, deleted or
  # newly created (untracked) must be committed together with the notebooks.
  out_of_sync = _git("diff", "--name-only", "--", str(PREVIEW_DIR))
  out_of_sync += _git(
      "ls-files", "--others", "--exclude-standard", "--", str(PREVIEW_DIR)
  )
  if not out_of_sync:
    return 0
  if not os.environ.get("CI"):
    # Local commit: add the previews to the commit that is being made.
    _git("add", "--all", "--", str(PREVIEW_DIR))
    print("Added these notebook previews to the commit:")
    for path in sorted(set(out_of_sync)):
      print(f"  {path}")
    return 0
  # CI only checks, so that a missing or stale preview fails the build.
  print("These notebook previews are out of date:")
  for path in sorted(set(out_of_sync)):
    print(f"  {path}")
  print(
      "Run `pre-commit install` (or `pre-commit run jupytext --all-files`)"
      f" and commit {PREVIEW_DIR}."
  )
  print(
      "If you did not change these notebooks, main is out of sync and"
      " committing the regenerated previews fixes it."
  )
  return 1


if __name__ == "__main__":
  sys.exit(main())
