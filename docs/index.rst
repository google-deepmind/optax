:github_url: https://github.com/deepmind/optax/tree/master/docs

Optax Documentation
===================


Optax is a gradient processing and optimization library for JAX.

Optax is designed to facilitate research by providing building blocks that can be easily recombined in custom ways.

Our goals are to:

- provide simple, well-tested, efficient implementations of core components,
- improve research productivity by enabling to easily combine low level ingredients into custom optimiser (or other gradient processing components).
- accelerate adoption of new ideas by making it easy for anyone to contribute.


We favour focusing on small composable building blocks that can be effectively
combined into custom solutions. Others may build upon these basic components
more complicated abstractions. Whenever reasonable, implementations prioritise
readability and structuring code to match standard equations, over code reuse.

An initial prototype of this library was made available in JAX's experimental
folder as `jax.experimental.optix`. Given the wide adoption across DeepMind of
optix, and after a few iterations on the API, optix was eventually moved out of
experimental as a standalone open-source library, renamed `optax`.


Installation
------------

See https://github.com/google/jax#pip-installation for instructions on
installing JAX.

We suggest installing the latest version of Optax by running::

    $ pip install git+https://github.com/deepmind/optax


.. toctree::
   :caption: API Documentation
   :maxdepth: 1

   api

Contribute
----------

- Issue tracker: https://github.com/deepmind/optax/issues
- Source code: https://github.com/deepmind/optax/tree/master

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/deepmind/optax/issues>`_.

License
-------

Optax is licensed under the Apache 2.0 License.

Indices and tables
==================

* :ref:`genindex`
