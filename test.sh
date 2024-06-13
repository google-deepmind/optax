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

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
rm -rf _testing
rm -rf dist/
rm -rf *.whl
rm -rf .pytype
mkdir -p _testing
readonly VENV_DIR="$(mktemp -d `pwd`/_testing/optax-env.XXXXXXXX)"
# in the unlikely case in which there was something in that directory
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install -q --upgrade pip setuptools wheel
pip install -q flake8 pytest-xdist pylint pylint-exit
pip install -q -e ".[test, examples]"

# Dp-accounting specifies exact minor versions as requirements which sometimes
# become incompatible with other libraries optax needs. We therefore install
# dependencies for dp-accounting manually.
# TODO(b/239416992): Remove this workaround if dp-accounting switches to minimum
# version requirements.
pip install -q -e ".[dp-accounting]"
pip install -q "dp-accounting>=0.1.1" --no-deps

# Ensure optax was not installed by one of the dependencies above,
# since if it is, the tests below will be run against that version instead of
# the branch build.
pip uninstall -q -y optax || true

# Lint with flake8.
flake8 `find optax examples -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Download Google OSS config.
wget -nd -v -t 3 -O .pylintrc https://google.github.io/styleguide/pylintrc
# Append specific config lines.
echo "disable=unnecessary-lambda-assignment,no-value-for-parameter,use-dict-literal" >> .pylintrc
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find optax examples -name '*.py' | grep -v 'test.py' | xargs` -d E1102 || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find optax examples -name '*_test.py' | xargs` -d W0212,E1102 || pylint-exit $PYLINT_ARGS $?
# Cleanup.
rm .pylintrc

# Build the package.
pip install build
python -m build
pip wheel --verbose --no-deps --no-clean dist/optax*.tar.gz
pip install optax*.whl

# Check types with pytype.
pip install -q pytype
pytype `find optax/_src examples optax/contrib -name '*.py' | xargs` -k -d import-error

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
cd _testing
python -m pytest -n auto --pyargs optax
cd ..

# Build Sphinx docs.
pip install -q -e ".[docs]"
# NOTE(vroulet) We have dependencies issues:
# tensorflow > 2.13.1 requires ml-dtypes <= 0.3.2
# but jax requires ml-dtypes >= 0.4.0
# So the environment is solved with tensorflow == 2.13.1 which requires
# typing_extensions < 4.6, which in turn prevents the import of TypeAliasType in
# IPython. We solve it here by simply upgrading typing_extensions to avoid that
# bug (which issues conflict warnings but runs fine).
# A long term solution is probably to fully remove tensorflow from our
# dependencies.
pip install --upgrade -v typing_extensions
cd docs && make html
# run doctests
make doctest
cd ..

# cleanup
rm -rf _testing

set +u
deactivate
echo "All tests passed. Congrats!"
