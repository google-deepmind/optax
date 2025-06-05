# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Tests for optax.perturbations, checking values and gradients."""

from functools import partial  # pylint: disable=g-importing-member
import operator

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax.perturbations import _make_pert
import optax.tree
import itertools
import functools

_TEST_SEEDS = list(range(3))

def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
  """An argmax one-hot function for arbitrary shapes."""
  inputs_flat = jnp.reshape(inputs, (-1))
  flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
  return jnp.reshape(flat_one_hot, inputs.shape)


def argmax_tree(x):
  return jax.tree.map(one_hot_argmax, x)

class MakePertTest(parameterized.TestCase):

  @parameterized.product(xtype=['tree', 'array'], sigma=[0.5, 1.0, 2.0], seed=_TEST_SEEDS)
  def test_pert_argmax(self,xtype,sigma,seed):
    """Test that make_perturbed_fun matches theoretical expression of gradient.

    Applies to the case of an argmax. Includes a test for the gradients and the
    Hessian of a scalar loss.

    This test is probabilistic, and can fail with low probability. If this
    happens, try increasing num_samples or loosening the tolerances of the
    assertions.
    """

    num_samples = 1_000_000

    rng = jax.random.key(seed)
    if xtype == 'tree':
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng,5)
        x = ( jax.random.normal(rng1,(4,)),
            {'k1': jax.random.normal(rng2,(3,)), 'k2': jax.random.normal(rng3,(2,))},
            jax.random.normal(rng4,(2,)) )
    else:
        rng, rng1 = jax.random.split(rng)
        x = jax.random.normal(rng1, (20,))

    # pert_argmax_fun and its jacobian should be an unbiased estimator of
    # softmax_fun and its jacobian
    pert_argmax_fun = _make_pert.make_perturbed_fun(
        argmax_tree, num_samples, sigma
    )
    softmax_fun = lambda t = sigma: jax.tree.map(lambda x: jax.nn.softmax(x / sigma), t)

    expected = softmax_fun(x)
    got = pert_argmax_fun(x,rng)
    chex.assert_trees_all_equal_shapes(got, expected)
    chex.assert_trees_all_close(got, expected, atol=1e-1)

    expected = jax.jacobian(softmax_fun)(x)
    got = jax.jacobian(pert_argmax_fun)(x,rng)
    chex.assert_trees_all_equal_shapes(got, expected)
    chex.assert_trees_all_close(got, expected, atol=1e-1)

    def pert_loss(inputs, rng):
      pert_argmax = pert_argmax_fun(inputs, rng)
      return optax.tree.sum(jax.tree.map(lambda x: x**2 + jnp.cos(x), pert_argmax))

    def exact_loss(inputs):
      softmax = softmax_fun(inputs)
      return optax.tree.sum(jax.tree.map(lambda x: x**2 + jnp.cos(x), softmax))

    expected = jax.grad(exact_loss)(x)
    got = jax.grad(pert_loss)(x,rng)
    chex.assert_trees_all_equal_shapes(got, expected)
    chex.assert_trees_all_close(got, expected, atol=1e-1)

    # Do not check the hessian, as the second order derivative is currently not
    # implemented with Monte-Carlo methods, and there is no reason to expect correct
    # values

  @parameterized.product(seed=_TEST_SEEDS)
  def test_values_on_tree(self,seed):
    """Test that the perturbations are well applied for functions on trees.

    Checks that small perturbations on the inputs have small effects on values
    """
    rng = jax.random.key(seed)

    weight_shapes = [(13, 4), (8, 13)]
    biases_shapes = [(13,), (8,)]

    example_tree = []
    for i in range(2):
      example_tree.append({
          'weights': jnp.ones(weight_shapes[i]),
          'biases': jnp.ones(biases_shapes[i]),
      })

    element = jnp.array([1.0, 2.0, 3.0, 4.0])
    element_tree = [
        element,
        {'a': jnp.ones_like(element), 'b': jnp.zeros_like(element)},
    ]

    def apply_both(tree, element):
      x = element
      for i in range(2):
        s_tree = tree[i]
        x = jnp.dot(s_tree['weights'], x) + s_tree['biases']
      return x

    def apply_element_tree(tree):
      leaves, _ = jax.tree.flatten(tree)
      return_tree = jax.tree.map(partial(apply_both, tree), element_tree)
      return_tree.append(sum(jnp.sum(leaf) for leaf in leaves))
      return return_tree

    def loss(tree):
      pred = apply_element_tree(tree)
      pred_true = apply_element_tree(example_tree)
      tree_loss = jax.tree.map(lambda x, y: (x - y) ** 2, pred, pred_true)
      list_loss = jax.tree.reduce(operator.add, tree_loss)
      return jax.tree.map(lambda *leaves: sum(leaves) / len(leaves), list_loss)

    loss_pert = _make_pert.make_perturbed_fun(
        loss, num_samples=100, sigma=0.1, noise=_make_pert.Normal()
    )
    rngs = jax.random.split(rng, 3)
    low_loss = loss_pert(example_tree, rngs[0])
    high_loss = loss_pert(
        optax.tree.random_like(rngs[1], example_tree), rngs[2]
    )
    np.testing.assert_array_less(low_loss, high_loss)


  @parameterized.product(Ashape=[(3,2), (2,3)], sigma=[0.25, 1.0, 4.0], seed=_TEST_SEEDS)
  def test_pert_linear_function(self,Ashape,sigma,seed):
    """Test that the peturbed function and its derivative is accurate in the
    transparent case of a linear function.

    This test is probabilistic, and can fail with low probability. If this
    happens, try increasing num_samples or loosening the tolerances of the
    assertions.
    """

    num_samples = 1_000_000

    rng = jax.random.key(seed)
    m,n = Ashape
    rng, rng1, rng2, rng3 = jax.random.split(rng,4)
    A = jax.random.normal(rng1,(m,n))
    b = jax.random.normal(rng2,(m,))
    x = jax.random.normal(rng3,(n,))

    # pert_linear_fun and its jacobian should be an unbiased estimator of
    # linear_fun and its jacobian
    linear_fun = lambda x: A @ x + b
    pert_linear_fun = _make_pert.make_perturbed_fun(
            linear_fun, num_samples, sigma, _make_pert.Normal()
        )

    expected = linear_fun(x)
    got = pert_linear_fun(x,rng)
    chex.assert_trees_all_equal_shapes(got, expected)
    chex.assert_trees_all_close(got, expected, atol=1e-1)

    expected = jax.jacobian(linear_fun)(x)
    got = jax.jacobian(pert_linear_fun)(x,rng)
    chex.assert_trees_all_equal_shapes(got, expected)
    chex.assert_trees_all_close(got, expected, atol=1e-1)

  def test_nonflat_shape(self):
    """Test that https://github.com/google-deepmind/optax/issues/1309 is fixed
    """
    rng = jax.random.key(0)
    x = jax.numpy.ones((1,2))
    f = lambda x: x[0,0]
    fp = _make_pert.make_perturbed_fun(f, 10)
    jax.grad(fp)(x, rng)


if __name__ == '__main__':
  absltest.main()
