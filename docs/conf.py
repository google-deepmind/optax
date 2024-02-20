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
"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
import inspect
import os
import sys

# The following typenames are re-written for public-facing type annotations.
TYPE_REWRITES = [
    ('~optax._src.base.GradientTransformation', 'optax.GradientTransformation'),
    ('~optax._src.base.Params', 'optax.Params'),
    ('~optax._src.base.Updates', 'optax.Updates'),
    ('~optax._src.base.OptState', 'optax.OptState'),
    ('base.GradientTransformation', 'optax.GradientTransformation'),
    ('base.Params', 'optax.Params'),
    ('base.Updates', 'optax.Updates'),
    ('base.OptState', 'optax.OptState'),
]


def _add_annotations_import(path):
  """Appends a future annotations import to the file at the given path."""
  with open(path) as f:
    contents = f.read()
  if contents.startswith('from __future__ import annotations'):
    # If we run sphinx multiple times then we will append the future import
    # multiple times too.
    return

  assert contents.startswith('#'), (path, contents.split('\n')[0])
  with open(path, 'w') as f:
    # NOTE: This is subtle and not unit tested, we're prefixing the first line
    # in each Python file with this future import. It is important to prefix
    # not insert a newline such that source code locations are accurate (we link
    # to GitHub). The assertion above ensures that the first line in the file is
    # a comment so it is safe to prefix it.
    f.write('from __future__ import annotations  ')
    f.write(contents)


def _recursive_add_annotations_import():
  for path, _, files in os.walk('../optax/'):
    for file in files:
      if file.endswith('.py'):
        _add_annotations_import(os.path.abspath(os.path.join(path, file)))


def _monkey_patch_doc_strings():
  """Rewrite function signatures to match the public API.

  This is a bit of a dirty hack, but it helps ensure that the public-facing
  docs have the correct type names and crosslinks.

  Since all optax code lives in a `_src` directory, and since all function
  annotations use types within that private directory, the public facing
  annotations are given relative to private paths.

  This means that the normal documentation generation process does not give
  the correct import paths, and the paths it does give cannot cross link to
  other parts of the documentation.

  Do we really need to use the _src structure for optax?

  Note, class members are not fixed by this patch, only function
    parameters. We should find a way to genearlize this solution.
  """
  import sphinx_autodoc_typehints
  original_process_docstring = sphinx_autodoc_typehints.process_docstring

  def new_process_docstring(app, what, name, obj, options, lines):
    result = original_process_docstring(app, what, name, obj, options, lines)

    for i in range(len(lines)):
      l = lines[i]
      for before, after in TYPE_REWRITES:
        l = l.replace(before, after)
      lines[i] = l

    return result

  sphinx_autodoc_typehints.process_docstring = new_process_docstring


if 'READTHEDOCS' in os.environ:
  _recursive_add_annotations_import()
  _monkey_patch_doc_strings()

sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.path.abspath('ext'))

import optax
from sphinxcontrib import katex

# -- Project information -----------------------------------------------------

project = 'Optax'
copyright = '2021, DeepMind'  # pylint: disable=redefined-builtin
author = 'Optax Contributors'

# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
    'sphinx_autodoc_typehints',
    'coverage_check',
    'myst_nb',  # This is used for the .ipynb notebooks
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.collections'
]

# so we don't have to do the canonical imports on every doctest
doctest_global_setup = '''
import optax
import jax
import jax.numpy as jnp
'''

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'exclude-members': '__repr__, __str__, __weakref__',
}

# -- Options for sphinx-collections

collections = {
    'examples': {
        'driver': 'copy_folder',
        'source': '../examples/',
        'ignore': 'BUILD'
    }
}


# -- Options for sphinx-gallery ----------------------------------------------

sphinx_gallery_conf = {
    'examples_dirs': '_collections/examples',  # path to your example scripts
    'gallery_dirs': (
        '_collections/generated_examples/'
    ),  # path to where to save gallery generated output
    'ignore_pattern': r'_test\.py',  # no gallery for test of examples
    'doc_module': 'optax',
    'backreferences_dir': os.path.join('modules', 'generated')
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'

html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/google-deepmind/optax',
    'use_repository_button': True,     # add a "link to repository" button
    'navigation_with_keys': False,
}

html_logo = 'images/logo.svg'
html_favicon = 'images/favicon.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

# -- Options for myst -------------------------------------------------------
nb_execution_mode = 'force'
nb_execution_allow_errors = False
nb_execution_excludepatterns = [
    # slow examples
    'cifar10_resnet.ipynb',
    'adversarial_training.ipynb',
    'reduce_on_plateau.ipynb',
    'differentially_private_sgd.ipynb'
]

# -- Options for katex ------------------------------------------------------

# See: https://sphinxcontrib-katex.readthedocs.io/en/0.4.1/macros.html
latex_macros = r"""
    \def \d              #1{\operatorname{#1}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = (
    '{displayMode: true, fleqn: true, macros: {' + katex_macros + '}}'
)

# Add LaTeX macros for LATEX builder
latex_elements = {'preamble': latex_macros}

# -- Source code links -------------------------------------------------------


def linkcode_resolve(domain, info):
  """Resolve a GitHub URL corresponding to Python object."""
  if domain != 'py':
    return None

  try:
    mod = sys.modules[info['module']]
  except ImportError:
    return None

  obj = mod
  try:
    for attr in info['fullname'].split('.'):
      obj = getattr(obj, attr)
  except AttributeError:
    return None
  else:
    obj = inspect.unwrap(obj)

  try:
    filename = inspect.getsourcefile(obj)
  except TypeError:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except OSError:
    return None

  # TODO(slebedev): support tags after we release an initial version.
  return (
      'https://github.com/google-deepmind/optax/tree/main/optax/%s#L%d#L%d'
      % (
          os.path.relpath(filename, start=os.path.dirname(optax.__file__)),
          lineno,
          lineno + len(source) - 1,
      )
  )


# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

source_suffix = ['.rst', '.md', '.ipynb']
