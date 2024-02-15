# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Solver API."""

from typing import Any, NamedTuple, Protocol, Union

import optax

Params = optax.Params
SolverState = Any


class SolverInitFn(Protocol):
  """A callable type for the `init` function of a `Solver`.

  The `init` function takes a tree of `params` and uses these to construct an
  arbitrary structured initial `state` for the solver. This
  may hold statistics of the past updates or any other non static information.
  """

  def __call__(self, params: Params) -> SolverState:
    """Initialize the solver.

    Args:
      params: The initial value of the parameters.

    Returns:
      The initial state of the solver.
    """


class SolverStepFn(Protocol):
  """A callable type for the `step` function of a `Solver`.

  The `step` function takes a tree of candidate parameters `params`, and an
  arbitrary structured `state` to return a new tree of candidate parameters,
  and a new state. Additional arguments can be fed in a keyword format.
  """

  def __call__(
      self, params: Params, state: SolverState, **extra_kwargs: dict[str, Any]
  ) -> tuple[Union[Params, tuple[Params, Any]], SolverState]:
    """Performs a step of the solver.

    Args:
      params: A tree of candidate parameters.
      state: The state of the solver.
      **extra_kwargs: Additional arguments for the function or the solver in
        keyword format.

    Returns:
      The updated parameters, eventually with an auxiliary output,
      and updated state.
    """


class Solver(NamedTuple):
  """A pair of pure functions implementing a solver.

  The init function initializes the state of the solver given an initial tree of
  parameters. The step function updates the parameters and the state of the 
  solver given current parameters and state.
  Contrarily to GradientTransformation, this API accesses the function to be
  optimized directly to compute gradients, then update directions and finally
  updated parameters.

  Attributes:
    init: A pure function which, when called with an example instance of the
      parameters, returns an arbitrary structured initial `state`.
    step: A pure function which takes as input a tree of parameters, the
      previous solver state (which may have been initialized using the init
      function). The step function then returns the updated parameters,
      and a new solver state.
  """

  init_fn: SolverInitFn
  step_fn: SolverStepFn
