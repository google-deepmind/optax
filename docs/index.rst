:github_url: https://github.com/google-deepmind/optax/tree/main/docs

=====
Optax
=====

Optax is a gradient processing and optimization library for JAX. It is designed
to facilitate research by providing building blocks that can be recombined in
custom ways in order to optimize parametric models such as, but not limited to,
deep neural networks.

Our goals are to

*   Provide readable, well-tested, efficient implementations of core components,
*   Improve researcher productivity by making it possible to combine low level
    ingredients into custom optimizer (or other gradient processing components).
*   Accelerate adoption of new ideas by making it easy for anyone to contribute.

We favor focusing on small composable building blocks that can be effectively
combined into custom solutions. Others may build upon these basic components
more complicated abstractions. Whenever reasonable, implementations prioritize
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
See `here <https://github.com/jax-ml/jax?tab=readme-ov-file#installation>`_
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

   api/assignment
   api/optimizers
   api/transformations
   api/combining_optimizers
   api/optimizer_wrappers
   api/optimizer_schedules
   api/apply_updates
   api/perturbations
   api/projections
   api/losses
   api/stochastic_gradient_estimators
   api/utilities
   api/contrib
   api/experimental


Support
-------

If you encounter issues with this software, please let us know by filing an issue on our `issue tracker <https://github.com/google-deepmind/optax/issues>`_. We are also happy to receive bug fixes and other contributions. For more information of how to contribute, please see the :doc:`development guide <development>`.


License
-------

Optax is licensed under the `Apache 2.0 License <https://github.com/google-deepmind/optax/blob/main/LICENSE>`_.


Indices and Tables
==================

* :ref:`genindex`
