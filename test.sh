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

# Install dependencies.
python3 -m pip install --quiet --upgrade pip setuptools wheel
python3 -m pip install --quiet --upgrade flake8 pytest-xdist pylint pylint-exit
python3 -m pip install --quiet --editable ".[test, examples]"

# Dp-accounting specifies exact minor versions as requirements which sometimes
# become incompatible with other libraries optax needs. We therefore install
# dependencies for dp-accounting manually.
# TODO(b/239416992): Remove this workaround if dp-accounting switches to minimum
# version requirements.
python3 -m pip install --quiet --editable ".[dp-accounting]"
python3 -m pip install --quiet --no-deps "dp-accounting>=0.1.1"

# Install the requested JAX version
if [ -z "${JAX_VERSION-}" ]; then
  : # use version installed in requirements above
elif [ "$JAX_VERSION" = "newest" ]; then
  python3 -m pip install --quiet --upgrade jax jaxlib
else
  python3 -m pip install --quiet "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
fi

# Ensure optax was not installed by one of the dependencies above,
# since if it is, the tests below will be run against that version instead of
# the branch build.
python3 -m pip uninstall --quiet --yes optax

# Lint with flake8.
python3 -m flake8 --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Append specific config lines.
# Lint modules and tests separately.
python3 -m pylint --rcfile=.pylintrc $(find optax -name '*.py' | grep -v 'test.py' | xargs) -d E1102 || pylint-exit $PYLINT_ARGS $?
# Disable protected-access warnings for tests.
python3 -m pylint --rcfile=.pylintrc $(find optax -name '*_test.py' | xargs) -d W0212,E1102 || pylint-exit $PYLINT_ARGS $?

# Build the package.
python3 -m pip install --quiet build
python3 -m build
python3 -m pip wheel --no-deps dist/optax-*.tar.gz --wheel-dir "${TEMP_DIR}"
python3 -m pip install --quiet "${TEMP_DIR}/optax-"*.whl

# Check types with pytype.
python3 -m pip install --quiet pytype
pytype "optax" --keep-going --disable import-error

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
cd "${TEMP_DIR}"
python3 -m pytest --numprocesses auto --pyargs optax
cd "${REPO_DIR}"

# Build Sphinx docs.
python3 -m pip install --quiet --editable ".[docs]"
# NOTE(vroulet) We have dependencies issues:
# tensorflow > 2.13.1 requires ml-dtypes <= 0.3.2
# but jax requires ml-dtypes >= 0.4.0
# So the environment is solved with tensorflow == 2.13.1 which requires
# typing_extensions < 4.6, which in turn prevents the import of TypeAliasType in
# IPython. We solve it here by simply upgrading typing_extensions to avoid that
# bug (which issues conflict warnings but runs fine).
# A long term solution is probably to fully remove tensorflow from our
# dependencies.
python3 -m pip install --upgrade --verbose typing_extensions
cd docs
make html
make doctest # run doctests
cd ..

pip install -U ruff
ruff check .

echo "All tests passed. Congrats!"
