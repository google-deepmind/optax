:github_url: https://github.com/google-deepmind/optax/tree/main/docs

=====
Optax
=====

Optax is a gradient processing and optimization library for JAX. It is designed
to facilitate research by providing building blocks that can be recombined in
custom ways in order to optimise parametric models such as, but not limited to,
deep neural networks.

Our goals are to

*   Provide readable, well-tested, efficient implementations of core components,
*   Improve researcher productivity by making it possible to combine low level
    ingredients into custom optimiser (or other gradient processing components).
*   Accelerate adoption of new ideas by making it easy for anyone to contribute.

We favour focusing on small composable building blocks that can be effectively
combined into custom solutions. Others may build upon these basic components
more complicated abstractions. Whenever reasonable, implementations prioritise
readability and structuring code to match standard equations, over code reuse.

Installation
------------

The latest release of Optax can be installed from
`PyPI <https://pypi.org/project/optax/>`_ using::

   pip install optax

You may also install directly from GitHub, using the following command. This
can be used to obtain the most recent version of Optax::

   pip install git+git://github.com/google-deepmind/optax.git

Note that Optax is built on top of JAX.
See `here <https://github.com/google/jax?tab=readme-ov-file#installation>`_
for instructions on installing JAX.


.. toctree::
   :hidden:

   getting_started

   gallery

   development


.. toctree::
   :hidden:
   :caption: 📖 Reference
   :maxdepth: 2

   api/optimizers
   api/transformations
   api/combining_optimizers
   api/optimizer_wrappers
   api/optimizer_schedules
   api/apply_updates
   api/losses
   api/linear_algebra_operators
   api/control_variates
   api/stochastic_gradient_estimators
   api/utilities
   api/contrib


The Team
--------

The development of Optax is led by Ross Hemsley, Matteo Hessel, Markus Kunesch
and Iurii Kemaev. The team relies on outstanding contributions from Research
Engineers and Research Scientists from throughout
`Google DeepMind <https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/>`_
and Alphabet. We are also very grateful to Optax's open source community for
contributing ideas, bug fixes, issues, design docs, and amazing new features.

The work on Optax is part of a wider effort to contribute to making the
`JAX Ecosystem <https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/>`_
the best possible environment for ML/AI research.

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/google-deepmind/optax/issues>`_.


License
-------

Optax is licensed under the Apache 2.0 License.


Indices and Tables
==================

* :ref:`genindex`
