# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

function cleanup {
  deactivate
  rm -r "${TEMP_DIR}"
}
trap cleanup EXIT

REPO_DIR=$(pwd)
TEMP_DIR=$(mktemp --directory)

set -o errexit
set -o nounset
set -o pipefail

# Install deps in a virtual env.
python3 -m venv "${TEMP_DIR}/test_venv"
source "${TEMP_DIR}/test_venv/bin/activate"

# Run the linter first to check lint errors quickly
python3 -m pip install --quiet --upgrade pip uv
python3 -m uv pip install --quiet pre-commit
pre-commit run -a

# Install dependencies.
python3 -m uv pip install --quiet --upgrade pip setuptools wheel
python3 -m uv pip install --quiet --upgrade flake8 pytest-xdist pylint pylint-exit
python3 -m uv pip install --quiet --editable ".[test]"

# Install the requested JAX version
if [ -z "${JAX_VERSION-}" ]; then
  : # use version installed in requirements above
elif [ "$JAX_VERSION" = "newest" ]; then
  python3 -m uv pip install --quiet --upgrade jax jaxlib
elif [ "$JAX_VERSION" = "nightly" ]; then
  python3 -m uv pip install --quiet --upgrade --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
else
  python3 -m uv pip install --quiet "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
fi

# Ensure optax was not installed by one of the dependencies above,
# since if it is, the tests below will be run against that version instead of
# the branch build.
python3 -m uv pip uninstall optax

# Lint with flake8.
python3 -m flake8 --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
pylint .

# Build the package.
python3 -m uv pip install --quiet build
python3 -m build
python3 -m pip wheel --no-deps dist/optax-*.tar.gz --wheel-dir "${TEMP_DIR}"
python3 -m pip install --quiet "${TEMP_DIR}/optax-"*.whl

# Check types with pytype.
python3 -m pip install --quiet pytype
pytype "optax" -j auto --keep-going --disable import-error

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
cd "${TEMP_DIR}"
python3 -m pytest --numprocesses auto --pyargs optax
#python3 -m pytest --numprocesses 8 --pyargs optax
cd "${REPO_DIR}"

# Build Sphinx docs.
python3 -m uv pip install --quiet --editable ".[docs]"
cd docs
make html
make doctest # run doctests
cd ..

echo "All tests passed. Congrats!"
