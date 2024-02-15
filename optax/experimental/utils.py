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
"""Utilities for solvers.
"""
import functools
import inspect
import operator

from typing import Any, Callable, Sequence


def split_kwargs(
    funs: Sequence[Callable[..., Any]],
    fun_kwargs: dict[str, Any],
) -> Sequence[dict[str, Any]]:
  """Split fun_kwargs into kwargs of the input functions funs.

  Raises an error in one keyword argument of fun_kwargs does not match any
  argument name of funs.

  Args:
    funs: sequence of functions to feed fun_kwargs to
    fun_kwargs: dictionary of keyword variables to be fed to funs

  Returns:
    (fun_1_kwargs, ..., fun_n_kwargs): keyword arguments for each function taken
      from fun_kwargs.

  Examples:
    >>> def fun1(a, b): return a+b
    >>> def fun2(c, d): return c+d
    >>> fun_kwargs = {'b':1., 'd':2.}
    >>> funs_kwargs = split_kwargs((fun1, fun2), fun_kwargs)
    >>> print(funs_kwargs)
    [{'b': 1.0}, {'d': 2.0}]
  """
  funs_arg_names = [
      list(inspect.signature(fun).parameters.keys()) for fun in funs
  ]
  funs_kwargs = [
      {k: v for k, v in fun_kwargs.items() if k in fun_arg_names}
      for fun_arg_names in funs_arg_names
  ]
  all_possible_arg_names = functools.reduce(operator.add, funs_arg_names)
  remaining_keys = [
      k for k in fun_kwargs.keys() if k not in all_possible_arg_names
  ]
  if remaining_keys:
    raise ValueError(
        f'{remaining_keys} are not valid arguments for any of the functions'
        f' {funs}'
    )
  return funs_kwargs
