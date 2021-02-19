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
"""Flexibly compose gradient transformations."""

from optax._src import transform
GradientTransformation = transform.GradientTransformation


def chain(*args: GradientTransformation) -> GradientTransformation:
  """Applies a list of chainable update transformations.

  Given a sequence of chainable transforms, `chain` returns an `init_fn`
  that constructs a `state` by concatenating the states of the individual
  transforms, and returns an `update_fn` which chains the update transformations
  feeding the appropriate state to each.

  Args:
    *args: a sequence of chainable (init_fn, update_fn) tuples.

  Returns:
    A single (init_fn, update_fn) tuple.
  """

  init_fns, update_fns = zip(*args)

  def init_fn(params):
    return [fn(params) for fn in init_fns]

  def update_fn(updates, state, params=None):
    if len(update_fns) != len(state):
      raise ValueError('The number of updates and states has to be the same in '
                       'chain! Make sure you have called init first!')

    new_state = []
    for s, fn in zip(state, update_fns):
      updates, new_s = fn(updates, s, params)
      new_state.append(new_s)
    return updates, new_state

  return GradientTransformation(init_fn, update_fn)
