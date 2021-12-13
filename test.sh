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
readonly VENV_DIR=/tmp/optax-env
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install flake8 pytest-xdist pytype pylint pylint-exit
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-test.txt
pip install -r requirements/requirements-examples.txt

# Ensure optax was not installed by one of the dependencies above,
# since if it is, the tests below will be run against that version instead of
# the branch build.
pip uninstall -y optax || true

# Lint with flake8.
flake8 `find optax examples -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
# Fail on errors, warning, conventions and refactoring messages.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find optax examples -name '*.py' | grep -v 'test.py' | xargs` || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find optax examples -name '*_test.py' | xargs` -d W0212 || pylint-exit $PYLINT_ARGS $?

# Build the package.
python setup.py sdist
pip wheel --verbose --no-deps --no-clean dist/optax*.tar.gz
pip install optax*.whl

# Check types with pytype.
pytype `find optax/_src/ examples -name '*.py' | xargs` -k -d import-error

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
mkdir _testing && cd _testing
python -m pytest -n "$(grep -c ^processor /proc/cpuinfo)" --pyargs optax
cd ..

cd examples
python -m pytest -n "$(grep -c ^processor /proc/cpuinfo)" .
cd ..

# Build Sphinx docs.
pip install -r requirements/requirements-docs.txt
cd docs && make html
cd ..

set +u
deactivate
echo "All tests passed. Congrats!"
