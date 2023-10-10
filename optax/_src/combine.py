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
"""Flexibly compose gradient transformations."""

from typing import Callable, NamedTuple, Union, Mapping, Hashable

import jax

from optax._src import base
from optax._src import wrappers


def chain(
    *args: base.GradientTransformation,
) -> base.GradientTransformationExtraArgs:
  """Applies a list of chainable update transformations.

  Given a sequence of chainable transforms, `chain` returns an `init_fn`
  that constructs a `state` by concatenating the states of the individual
  transforms, and returns an `update_fn` which chains the update transformations
  feeding the appropriate state to each.

  Args:
    *args: a sequence of chainable (init_fn, update_fn) tuples.

  Returns:
    A ``GradientTransformationExtraArgs``, created by chaining the input
    transformations. Note that independent of the argument types, the resulting
    transformation always supports extra args. Any extra arguments passed to the
    returned transformation will be passed only to those transformations in the
    chain that support extra args.
  """

  transforms = [base.with_extra_args_support(t) for t in args]
  init_fns, update_fns = zip(*transforms)

  def init_fn(params):
    return tuple(fn(params) for fn in init_fns)

  def update_fn(updates, state, params=None, **extra_args):
    if len(update_fns) != len(state):
      raise ValueError('The number of updates and states has to be the same in '
                       'chain! Make sure you have called init first!')

    new_state = []
    for s, fn in zip(state, update_fns):
      updates, new_s = fn(updates, s, params, **extra_args)
      new_state.append(new_s)
    return updates, tuple(new_state)

  # We opt to always return the GradientTransformationExtraArgs type here,
  # instead of selecting the return type based on the arguments, since it works
  # much better with the currently available type checkers. It also means that
  # users will not get unexpected signature errors if they remove all of the
  # transformations in a chain accepting extra args.
  return base.GradientTransformationExtraArgs(init_fn, update_fn)


class MultiTransformState(NamedTuple):
  inner_states: Mapping[Hashable, base.OptState]


def multi_transform(
    transforms: Mapping[Hashable, base.GradientTransformation],
    param_labels: Union[base.PyTree, Callable[[base.PyTree], base.PyTree]]
) -> base.GradientTransformationExtraArgs:
  """Partitions params and applies a different transformation to each subset.

  Below is an example where we apply Adam to the weights and SGD to the biases
  of a 2-layer neural network::

    import optax
    import jax
    import jax.numpy as jnp

    def map_nested_fn(fn):
      '''Recursively apply `fn` to the key-value pairs of a nested dict'''
      def map_fn(nested_dict):
        return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()}
      return map_fn

    params = {'linear_1': {'w': jnp.zeros((5, 6)), 'b': jnp.zeros(5)},
              'linear_2': {'w': jnp.zeros((6, 1)), 'b': jnp.zeros(1)}}
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)  # dummy gradients

    label_fn = map_nested_fn(lambda k, _: k)
    tx = optax.multi_transform({'w': optax.adam(1.0), 'b': optax.sgd(1.0)},
                               label_fn)
    state = tx.init(params)
    updates, new_state = tx.update(gradients, state, params)
    new_params = optax.apply_updates(params, updates)

  Instead of providing a ``label_fn``, you may provide a PyTree of labels
  directly.  Also, this PyTree may be a prefix of the parameters PyTree. This
  is demonstrated in the GAN pseudocode below::

    generator_params = ...
    discriminator_params = ...
    all_params = (generator_params, discriminator_params)
    param_labels = ('generator', 'discriminator')

    tx = optax.multi_transform(
        {'generator': optax.adam(0.1), 'discriminator': optax.adam(0.5)},
        param_labels)

  If you would like to not optimize some parameters, you may wrap
  ``optax.multi_transform`` with :func:`optax.masked`.

  Args:
    transforms: A mapping from labels to transformations. Each transformation
      will be only be applied to parameters with the same label.
    param_labels: A PyTree that is the same shape or a prefix of the
      parameters/updates (or a function that returns one given the parameters as
      input). The leaves of this PyTree correspond to the keys of the transforms
      (therefore the values at the leaves must be a subset of the keys).

  Returns:
    An ``optax.GradientTransformation``.
  """

  transforms = {
      k: base.with_extra_args_support(v)
      for k, v in transforms.items()
  }

  def make_mask(labels, group):
    return jax.tree_util.tree_map(lambda label: label == group, labels)

  def init_fn(params):
    labels = param_labels(params) if callable(param_labels) else param_labels

    label_set = set(jax.tree_util.tree_leaves(labels))
    if not label_set.issubset(transforms.keys()):
      raise ValueError('Some parameters have no corresponding transformation.\n'
                       f'Parameter labels: {list(sorted(label_set))} \n'
                       f'Transforms keys: {list(sorted(transforms.keys()))} \n')

    inner_states = {
        group: wrappers.masked(tx, make_mask(labels, group)).init(params)
        for group, tx in transforms.items()
    }
    return MultiTransformState(inner_states)

  def update_fn(updates, state, params=None, **extra_args):
    labels = param_labels(updates) if callable(param_labels) else param_labels
    new_inner_state = {}
    for group, tx in transforms.items():
      masked_tx = wrappers.masked(tx, make_mask(labels, group))
      updates, new_inner_state[group] = masked_tx.update(
          updates, state.inner_states[group], params, **extra_args)
    return updates, MultiTransformState(new_inner_state)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
