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

set -o nounset

echo "Creating virtual environment"
python3 -m venv test_venv

echo "Activating virtual environment"
source test_venv/bin/activate

for dep in pip setuptools wheel flake8 pytest-xdist pylint pylint-exit pytype build ruff typing_extensions ".[test,examples,docs]"
do
  echo "Installing" $dep
  python3 -m pip install --quiet --upgrade $dep
done

# Dp-accounting specifies exact minor versions as requirements which sometimes
# become incompatible with other libraries optax needs. We therefore install
# dependencies for dp-accounting manually.
# TODO(b/239416992): Remove this workaround if dp-accounting switches to minimum
# version requirements.
python3 -m pip install --quiet --editable ".[dp-accounting]"
python3 -m pip install --quiet --no-deps "dp-accounting>=0.1.1"

echo "Installing requested JAX version: ${JAX_VERSION-}"
if [ -z "${JAX_VERSION-}" ]
then
  : # use version installed in requirements above
elif [ "$JAX_VERSION" = "newest" ]
then
  python3 -m pip install --quiet --upgrade jax jaxlib
elif [ "$JAX_VERSION" = "nightly" ]
then
  python3 -m pip install --quiet --upgrade --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
else
  python3 -m pip install --quiet "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
fi

# Ensure optax was not installed by one of the dependencies above,
# since if it is, the tests below will be run against that version instead of
# the branch build.
echo "Uninstalling optax (if installed)"
python3 -m pip uninstall --quiet --yes optax

echo "Linting with flake8"
flake8 --select=E9,F63,F7,F82,E225,E251 --show-source --statistics --exclude=build,dist,test_venv

echo "Linting with pylint"
pylint .

echo "Building package"
python3 -m build

echo "Building wheel"
python3 -m pip wheel --no-deps dist/optax-*.tar.gz

echo "Installing wheel"
python3 -m pip install --quiet optax-*.whl

echo "Type checking with pytype"
pytype

echo "Running tests with pytest"
pytest

echo "Building Sphinx docs"
# NOTE(vroulet) We have dependencies issues:
# tensorflow > 2.13.1 requires ml-dtypes <= 0.3.2
# but jax requires ml-dtypes >= 0.4.0
# So the environment is solved with tensorflow == 2.13.1 which requires
# typing_extensions < 4.6, which in turn prevents the import of TypeAliasType in
# IPython. We solve it here by simply upgrading typing_extensions to avoid that
# bug (which issues conflict warnings but runs fine).
# A long term solution is probably to fully remove tensorflow from our
# dependencies.
cd docs
make html

echo "Running doctests"
make doctest
cd ..

echo "Running ruff"
ruff check .

echo "All tests passed. Congrats!"
