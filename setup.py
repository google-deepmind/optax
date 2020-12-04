# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Install script for setuptools."""

from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  with open('optax/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=') + 1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `optax/__init__.py`')


setup(
    name='optax',
    version=_get_version(),
    url='https://github.com/deepmind/optax',
    license='Apache 2.0',
    author='DeepMind',
    description=('A gradient processing and optimisation library in JAX.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='optax-dev@google.com',
    keywords='reinforcement-learning python machine learning',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=[
        'absl-py>=0.7.1',
        'chex>=0.0.3',
        'jax>=0.1.55',
        'jaxlib>=0.1.37',
        'numpy>=1.18.0',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
