# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import test_utils
from optax.experimental import microbatching


def per_example_function(nonbatch_arg, batch_arg1, batch_arg2):
  return batch_arg1 * batch_arg2, batch_arg1 / batch_arg2 * nonbatch_arg


def sum_function(nonbatch_arg, batch_arg1, batch_arg2):
  return jnp.sum(nonbatch_arg + batch_arg1 + 2 * batch_arg2, axis=0)


def mean_function(nonbatch_arg, batch_arg1, batch_arg2):
  return jnp.mean(nonbatch_arg * batch_arg1), {'key': jnp.mean(batch_arg2)}


def sum_function_with_kwargs(nonbatch_arg, batch_arg1, *, batch_kwarg2):
  return jnp.sum(nonbatch_arg + batch_arg1 + 2 * batch_kwarg2, axis=0)


def sum_mean_per_example_function(nonbatch_arg, batch_arg1, batch_arg2):
  return {
      'per_example': per_example_function(nonbatch_arg, batch_arg1, batch_arg2),
      'sum': sum_function(nonbatch_arg, batch_arg1, batch_arg2),
      'mean': mean_function(nonbatch_arg, batch_arg1, batch_arg2),
  }


NONBATCH_SHAPE = ()
BATCH_SHAPE = (10, 3)


FUNCTION_ACCUM_PAIRS = {
    'per_example': (
        per_example_function,
        microbatching.AccumulationType.CONCAT,
    ),
    'sum': (sum_function, microbatching.AccumulationType.SUM),
    'mean': (mean_function, microbatching.AccumulationType.MEAN),
    'kwarg': (sum_function_with_kwargs, microbatching.AccumulationType.SUM),
    'sum_mean_per_example': (
        sum_mean_per_example_function,
        {
            'per_example': microbatching.AccumulationType.CONCAT,
            'sum': microbatching.AccumulationType.SUM,
            'mean': microbatching.AccumulationType.MEAN,
        },
    ),
}


class MicrobatchingTest(parameterized.TestCase):

  @parameterized.product(
      name=['per_example', 'sum', 'mean', 'sum_mean_per_example'],
      microbatch_size=[1, 2, 5, 10],
  )
  def test_microbatch(self, name: str, microbatch_size: int):
    nonbatch_arg = jnp.array(np.random.normal(size=NONBATCH_SHAPE))
    batch_arg1 = jnp.array(np.random.normal(size=BATCH_SHAPE))
    batch_arg2 = jnp.array(np.random.normal(size=BATCH_SHAPE))
    fun, accumulator = FUNCTION_ACCUM_PAIRS[name]
    microbatched_fun = microbatching.microbatch(
        fun,
        argnums=(1, 2),
        microbatch_size=microbatch_size,
        accumulator=accumulator,
    )
    expected_answer = fun(nonbatch_arg, batch_arg1, batch_arg2)
    actual_answer = microbatched_fun(nonbatch_arg, batch_arg1, batch_arg2)
    test_utils.assert_trees_all_close(expected_answer, actual_answer, atol=1e-6)

  def test_microbatch_with_kwargs(self):
    nonbatch_arg = jnp.array(np.random.normal(size=NONBATCH_SHAPE))
    batch_arg1 = jnp.array(np.random.normal(size=BATCH_SHAPE))
    batch_kwarg2 = jnp.array(np.random.normal(size=BATCH_SHAPE))
    fun, accumulator = FUNCTION_ACCUM_PAIRS['kwarg']
    microbatched_fun = microbatching.microbatch(
        fun,
        argnums=(1,),
        microbatch_size=2,
        accumulator=accumulator,
        argnames=('batch_kwarg2',),
    )
    expected_answer = fun(nonbatch_arg, batch_arg1, batch_kwarg2=batch_kwarg2)
    actual_answer = microbatched_fun(
        nonbatch_arg, batch_arg1, batch_kwarg2=batch_kwarg2
    )
    test_utils.assert_trees_all_close(expected_answer, actual_answer, atol=1e-6)

  def test_nondecomposable_function(self):
    def fun(x):
      return x[:-1] @ x[1:]

    x = jnp.arange(6).astype(jnp.float32)
    microbatched_fun = microbatching.microbatch(
        fun,
        argnums=0,
        microbatch_size=3,
        accumulator=microbatching.AccumulationType.SUM,
    )
    test_utils.assert_trees_all_close(
        fun(x), 0 * 1 + 1 * 2 + 2 * 3 + 3 * 4 + 4 * 5)
    # This split here is an implementation detail that could change:
    # microbatch1 = [0, 2, 4]
    # microbatch2 = [1, 3, 5]
    test_utils.assert_trees_all_close(
        microbatched_fun(x), 0 * 2 + 2 * 4 + 1 * 3 + 3 * 5
    )

  def test_raises_on_invalid_microbatches(self):
    nonbatch_arg = jnp.array(np.random.normal(size=NONBATCH_SHAPE))
    batch_arg1 = jnp.array(np.random.normal(size=BATCH_SHAPE))
    batch_arg2 = jnp.array(np.random.normal(size=BATCH_SHAPE))
    fun, accumulator = FUNCTION_ACCUM_PAIRS['sum']
    microbatched_fun = microbatching.microbatch(
        fun,
        argnums=(1, 2),
        microbatch_size=3,
        accumulator=accumulator,
    )
    with self.assertRaisesRegex(ValueError, 'divisible'):
      _ = microbatched_fun(nonbatch_arg, batch_arg1, batch_arg2)

  @parameterized.product(
      arg_dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
  )
  def test_correct_dtype_returned(self, arg_dtype):
    nonbatch_arg = jnp.array(np.random.normal(size=NONBATCH_SHAPE), arg_dtype)
    batch_arg1 = jnp.array(np.random.normal(size=BATCH_SHAPE), arg_dtype)
    batch_arg2 = jnp.array(np.random.normal(size=BATCH_SHAPE), arg_dtype)
    fun, accumulator = FUNCTION_ACCUM_PAIRS['sum']
    microbatched_fun = microbatching.microbatch(
        fun,
        argnums=(1, 2),
        microbatch_size=2,
        accumulator=accumulator,
    )
    answer = microbatched_fun(nonbatch_arg, batch_arg1, batch_arg2)
    self.assertEqual(answer.dtype, arg_dtype)

  def test_early_stopping_concat(self):
    x = jnp.arange(12).astype(jnp.float32) + 1

    output = microbatching.microbatch(
        lambda x: x,
        argnums=0,
        accumulator=microbatching.AccumulationType.CONCAT,
        microbatch_size=3,
        num_real_microbatches=2,
    )(x)

    test_utils.assert_trees_all_close(jnp.sum(output != 0), 6)

  @parameterized.parameters(
      microbatching.AccumulationType.SUM,
      microbatching.AccumulationType.MEAN,
      microbatching.AccumulationType.RUNNING_MEAN,
      microbatching.AccumulationType.CONCAT,
  )
  def test_in_axes_invariant(self, acc):

    arg_axis0 = jnp.array(np.random.normal(size=(10, 4, 5)))
    arg_axis1 = jnp.transpose(arg_axis0, axes=(1, 0, 2))
    self.assertEqual(arg_axis1.shape, (4, 10, 5))
    fun_axis0 = functools.partial(jnp.einsum, 'bij,bkj->ik')
    fun_axis1 = functools.partial(jnp.einsum, 'ibj,kbj->ik')

    result0 = microbatching.microbatch(
        fun_axis0, argnums=(0, 1), microbatch_size=2, in_axes=0, accumulator=acc
    )(arg_axis0, arg_axis0)
    result1 = microbatching.microbatch(
        fun_axis1, argnums=(0, 1), microbatch_size=2, in_axes=1, accumulator=acc
    )(arg_axis1, arg_axis1)
    test_utils.assert_trees_all_close(result0, result1)

  @parameterized.parameters(
      microbatching.AccumulationType.SUM,
      microbatching.AccumulationType.MEAN,
      microbatching.AccumulationType.RUNNING_MEAN,
      microbatching.AccumulationType.CONCAT,
  )
  def test_in_axes_with_argnames(self, acc):

    arg0 = jnp.array(np.random.normal(size=(10, 4, 5)))
    arg1 = jnp.transpose(arg0, axes=(1, 0, 2))
    self.assertEqual(arg1.shape, (4, 10, 5))

    def fun(a, b):
      return jnp.einsum('bij,kbj->ik', a, b)

    mfun = functools.partial(
        microbatching.microbatch, fun, microbatch_size=2, accumulator=acc
    )

    ans0 = mfun(argnums=(0, 1), in_axes=(0, 1))(arg0, arg1)
    ans1 = mfun(argnums=(1, 0), in_axes=(1, 0))(arg0, arg1)
    ans2 = mfun(argnums=(), argnames=('a', 'b'), in_axes=(0, 1))(a=arg0, b=arg1)
    ans3 = mfun(argnums=(), argnames=('a', 'b'), in_axes=(0, 1))(b=arg1, a=arg0)
    ans4 = mfun(argnums=(), argnames=('b', 'a'), in_axes=(1, 0))(a=arg0, b=arg1)
    ans5 = mfun(argnums=(), argnames=('b', 'a'), in_axes=(1, 0))(b=arg1, a=arg0)
    ans6 = mfun(argnums=0, argnames='b', in_axes=(0, 1))(arg0, b=arg1)

    test_utils.assert_trees_all_close(ans0, ans1)
    test_utils.assert_trees_all_close(ans0, ans2)
    test_utils.assert_trees_all_close(ans0, ans3)
    test_utils.assert_trees_all_close(ans0, ans4)
    test_utils.assert_trees_all_close(ans0, ans5)
    test_utils.assert_trees_all_close(ans0, ans6)

  def test_argnums_argnames_invariant(self):
    def fun(a, b, c, *, d, e, f):
      return jnp.sum(a + b + c + d + e + f)

    output1 = microbatching.microbatch(
        fun, argnums=(0, 1), microbatch_size=2,
    )(jnp.ones(16), jnp.ones(16), 1, d=2, e=3, f=4)

    output2 = microbatching.microbatch(
        fun, argnums=0, argnames='b', microbatch_size=2,
    )(jnp.ones(16), b=jnp.ones(16), c=1, d=2, e=3, f=4)

    test_utils.assert_trees_all_close(output1, output2)

    output3 = microbatching.microbatch(
        fun, argnums=(), argnames=('b', 'a'), microbatch_size=2,
    )(a=jnp.ones(16), b=jnp.ones(16), c=1, d=2, e=3, f=4)

    test_utils.assert_trees_all_close(output1, output3)

  def test_vmap(self):
    x = jnp.arange(2*4*8).reshape(2, 4, 8)
    custom_vmap = microbatching.micro_vmap(
        jnp.sum,
        in_axes=1,
        microbatch_size=2,
    )
    normal_vmap = jax.vmap(jnp.sum, in_axes=1)
    test_utils.assert_trees_all_close(normal_vmap(x), custom_vmap(x), atol=1e-6)

  @parameterized.parameters([False, True])
  def test_micro_grad_basic(self, keep_batch_dim):
    # This function definition is valid whether or not features/targets has a
    # batch axis or not (shape (1,) or shape (B,1)).
    def mean_squared_loss(params, features, targets):
      assert features.ndim == int(keep_batch_dim) + 1
      preds = features @ params
      diff = preds - targets
      return 0.5 * jnp.mean(diff**2)

    params = jnp.zeros(1)
    features = jnp.ones((4, 1))
    targets = jnp.array([0, 1, 2, 3], dtype=jnp.float32)

    grad_fn = microbatching.micro_grad(
        mean_squared_loss,
        argnums=0,
        batch_argnums=(1, 2),
        transform_fn=lambda x: (x, x**2),
        metrics_fn=jnp.linalg.norm,
        keep_batch_dim=keep_batch_dim,
    )

    (grads, squared_grads), aux = grad_fn(params, features, targets)
    expected_grad = -(0 + 1 + 2 + 3)
    expected_squared_grad = 0**2 + 1**2 + 2**2 + 3**2
    expected_values = jnp.array([0, 0.5, 2, 4.5], dtype=jnp.float32)
    expected_norms = jnp.array([0, 1, 2, 3], dtype=jnp.float32)

    test_utils.assert_trees_all_close(grads[0], expected_grad)
    test_utils.assert_trees_all_close(squared_grads[0], expected_squared_grad)
    test_utils.assert_trees_all_close(aux.values, expected_values)
    test_utils.assert_trees_all_close(aux.metrics, expected_norms)
    self.assertIsNone(aux.aux)


if __name__ == '__main__':
  absltest.main()
