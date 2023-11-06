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
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install flake8 pytest-xdist pylint pylint-exit
pip install -e ".[test, examples]"

# Dp-accounting specifies exact minor versions as requirements which sometimes
# become incompatible with other libraries optax needs. We therefore install
# dependencies for dp-accounting manually.
# TODO(b/239416992): Remove this workaround if dp-accounting switches to minimum
# version requirements.
pip install -e ".[dp-accounting]"
pip install "dp-accounting>=0.1.1" --no-deps

# Ensure optax was not installed by one of the dependencies above,
# since if it is, the tests below will be run against that version instead of
# the branch build.
pip uninstall -y optax || true

# Lint with flake8.
flake8 `find optax docs/examples -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
PYLINT_ARGS="-efail -wfail -cfail -rfail"
# Download Google OSS config.
wget -nd -v -t 3 -O .pylintrc https://google.github.io/styleguide/pylintrc
# Append specific config lines.
echo "disable=unnecessary-lambda-assignment,no-value-for-parameter,use-dict-literal" >> .pylintrc
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find optax docs/examples -name '*.py' | grep -v 'test.py' | xargs` || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find optax docs/examples -name '*_test.py' | xargs` -d W0212 || pylint-exit $PYLINT_ARGS $?
# Cleanup.
rm .pylintrc

# Build the package.
pip install build
python -m build
pip wheel --verbose --no-deps --no-clean dist/optax*.tar.gz
pip install optax*.whl

# Check types with pytype.
# Note: pytype does not support 3.11 as of 25.06.23
# See https://github.com/google/pytype/issues/1308
if [ `python -c 'import sys; print(sys.version_info.minor)'` -lt 11 ];
then
  pip install pytype
  pytype `find optax/_src/ docs/examples -name '*.py' | xargs` -k -d import-error
fi;

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
mkdir _testing && cd _testing
python -m pytest -n auto --pyargs optax
cd ..

cd docs/examples
python -m pytest -n auto .
# remove __pycache__ directories created by pytest
rm -rf __pycache__
cd ../..

# Build Sphinx docs.
pip install -e ".[docs]"
cd docs && make html
cd ..

set +u
deactivate
echo "All tests passed. Congrats!"
