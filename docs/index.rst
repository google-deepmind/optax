:github_url: https://github.com/deepmind/optax/tree/master/docs

Optax
-----

Optax is a gradient processing and optimization library for JAX.

Optax is designed to facilitate research by providing building blocks
that can be easily recombined in custom ways.

Our goals are to:

*   provide readable, well-tested, efficient implementations of core components,
*   improve researcher productivity by making it possible to combine low level
    ingredients into custom optimiser (or other gradient processing components).
*   accelerate adoption of new ideas by making it easy for anyone to contribute.

We favour focusing on small composable building blocks that can be effectively
combined into custom solutions. Others may build upon these basic components
more complicated abstractions. Whenever reasonable, implementations prioritise
readability and structuring code to match standard equations, over code reuse.

An initial prototype of this library was made available in JAX's experimental
folder as `jax.experimental.optix`. Given the wide adoption across DeepMind
of `optix`, and after a few iterations on the API, `optix` was eventually moved
out of `experimental` as a standalone open-source library, renamed `optax`.

Installation
------------

The latest release of Optax can be installed from PyPI by simply running:

```shell
pip install optax
```

You may also instal directly from github's head, using the following command:

```shell
pip install git+git://github.com/deepmind/optax.git
```

Note that Optax is built on top of JAX.
See `here <https://github.com/google/jax#pip-installation-cpu>`_
for instructions on installing JAX.


.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   optax-101


.. toctree::
   :caption: Examples
   :maxdepth: 1

   meta_learning


.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api


Contribute
----------

We'd love to accept your patches and contributions to this project. Please take
a look at the simple guidelines in the
`CONTRIBUTING.md https://github.com/deepmind/optax/blob/master/CONTRIBUTING.md`_
file before starting your first PR. You can find open issues and the source code
in the links below.

- `Issue tracker <https://github.com/deepmind/optax/issues>`_
- `Source code <https://github.com/deepmind/optax/tree/master>`_

A selection of good first issues for new contributors are labelled accordingly
in the issue tracker. You are also welcome to create new issues. For large
changes that warrant extensive discussion of the implications for all users
consider creating a Design doc, as done by @wdphy16 in this example:

- `Complex number support - https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29`_

If in doubt whether or not a proposed change deserves its own design doc,
just start with opening the issue and we can discuss there.

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/deepmind/optax/issues>`_.

License
-------

Optax is licensed under the Apache 2.0 License.


Indices and Tables
==================

* :ref:`genindex`
