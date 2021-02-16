# Optax

![CI status](https://github.com/deepmind/optax/workflows/tests/badge.svg)

## Introduction

Optax is a gradient processing and optimization library for JAX.

Optax is designed to facilitate research by providing building blocks
that can be easily recombined in custom ways.

Our goals are to:

*   provide simple, well-tested, efficient implementations of core components,
*   improve research productivity by enabling to easily combine low level
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

## Installation

Optax can be installed with pip directly from github, with the following command:

`pip install git+git://github.com/deepmind/optax.git`

or from PyPI:

`pip install optax`

## Components

### Gradient Transformations ([transform.py](https://github.com/deepmind/optax/blob/master/optax/_src/transform.py))

One of the key building blocks of `optax` is a `GradientTransformation`.

Each transformation is defined two functions:

*   state = init(params)
*   grads, state = update(grads, state, params=None)

The `init` function initializes a (possibly empty) set of statistics (aka state)
and the `update` function transforms a candidate gradient given some statistics,
and (optionally) the current value of the parameters.

For example:

```python
tx = scale_by_rms()
state = tx.init(params)  # init stats
grads, state = tx.update(grads, state, params)  # transform & update stats.
```

### Composing Gradient Transformations ([combine.py](https://github.com/deepmind/optax/blob/master/optax/_src/combine.py))

The fact that transformations take candidate gradients as input and return
processed gradients as output (in contrast to returning the updated parameters)
is critical to allow to combine arbitrary transformations into a custom
optimiser / gradient processor, and also allows to combine transformations for
different gradients that operate on a shared set of variables.

For instance, `chain` combines them sequentially, and returns a
new `GradientTransformation` that applies several transformations in sequence.

For example:

```python
my_optimiser = chain(
    clip_by_global_norm(max_norm),
    scale_by_adam(eps=1e-4),
    scale(-learning_rate))
```

### Schedules ([schedule.py](https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py))

Many popular transformations use time dependent components, e.g. to anneal
some hyper-parameter (e.g. the learning rate). Optax provides for this purpose
`schedules` that can be used to decay scalars as a function of a `step` count.

For example you many use a polynomial schedule (with `power=1`) to decay
a hyper-parameter linearly over a number of steps:

```python
schedule_fn = polynomial_schedule(
    init_value=1., end_value=0., power=1, transition_steps=5)

for step_count in range(6):
  print(schedule_fn(step_count))  # [1., 0.8, 0.6, 0.4, 0.2, 0.]
```

Schedules are used by certain gradient transformation, for instance:

```python
schedule_fn = polynomial_schedule(
    init_value=-learning_rate, end_value=0., power=1, transition_steps=5)
optimiser = chain(
    clip_by_global_norm(max_norm),
    scale_by_adam(eps=1e-4),
    scale_by_schedule(schedule_fn))
```

### Popular optimisers ([alias.py](https://github.com/deepmind/optax/blob/master/optax/_src/alias.py))

In addition to the low level building blocks we also provide aliases for popular
optimisers built using these components (e.g. RMSProp, Adam, AdamW, etc, ...).
These are all still instances of a `GradientTransformation`, and can therefore
be further combined with any of the individual building blocks.

For example:

```python
def adamw(learning_rate, b1, b2, eps, weight_decay):
  return chain(
      scale_by_adam(b1=b1, b2=b2, eps=eps),
      scale_and_decay(-learning_rate, weight_decay=weight_decay))
```

### Applying updates ([update.py](https://github.com/deepmind/optax/blob/master/optax/_src/update.py))

An `apply_updates` function can be used to eventually apply the
transformed gradients to the set of parameters of interest.

```python
updates, state = tx.update(grads, state, params)  # transform & update stats.
new_params = optax.apply_updates(params, updates)  # update the parameters.
```

Note that separating gradient transformations from the parameter update is
critical to support composing sequence of transformations (e.g. `chain`), as
well as combine multiple updates to the same parameters (e.g. in multi-task
settings where different tasks need different sets of gradient transformations).

### Second Order ([second_order.py](https://github.com/deepmind/optax/blob/master/optax/_src/second_order.py))

Computing the Hessian or Fisher information matrices for neural networks is
typically intractable due to the quadratic memory requirements. Solving for the
diagonals of these matrices is often a better solution. The library offers
functions for computing these diagonals with sub-quadratic memory requirements.

### Stochastic gradient estimators ([stochastic_gradient_estimators.py](https://github.com/deepmind/optax/blob/master/optax/_src/stochastic_gradient_estimators.py))

Stochastic gradient estimators compute Monte Carlo estimates of gradients of
the expectation of a function under a distribution with respect to the
distribution's parameters.

Unbiased estimators such as the score function estimator (REINFORCE),
pathwise estimator (reparametrization trick) or measure valued estimator
are implemented: `score_function_jacobians`, `pathwise_jacobians` and `
measure_valued_jacobians`. Their applicability (both in terms of functions and
distributions) is discussed in their respective documentation.

Stochastic gradient estimators can be combined with common control variates for
variance reduction via `control_variates_jacobians`. For provided control
variates see `delta` and `moving_avg_baseline`.

The result of a gradient estimator or `control_variates_jacobians` contains the
Jacobians of the function with respect to the samples from the input
distribution. These can then be used to update distributional parameters, or
to assess gradient variance.

Example of how to use the `pathwise_jacobians` estimator:
```python
  dist_params = [mean, log_scale]
  function = lambda x: jnp.sum(x * weights)
  jacobians = pathwise_jacobians(
        function, dist_params,
        utils.multi_normal, rng, num_samples)

  mean_grads = jnp.mean(jacobians[0], axis=0)
  log_scale_grads = jnp.mean(jacobians[1], axis=0)
  grads = [mean_grads, log_scale_grads]
  optim_update, optim_state = optim.update(grads, optim_state)
  updated_dist_params = optax.apply_updates(dist_params, optim_update)

```

where `optim` is an optax optimizer.

## Citing Optax

To cite this repository:

```
@software{optax2020github,
  author = {Matteo Hessel and David Budden and Fabio Viola and Mihaela Rosca
            and Eren Sezener and Tom Hennigan},
  title = {Optax: composable gradient transformation and optimisation, in JAX!},
  url = {http://github.com/deepmind/optax},
  version = {0.0.1},
  year = {2020},
}
```

In this bibtex entry, the version number is intended to be from
[optax/\_\_init\_\_.py](https://github.com/deepmind/optax/blob/master/optax/__init__.py),
and the year corresponds to the project's open-source release.
