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
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax.perturbations import _make_pert
import optax.tree
import itertools
import functools


def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
  """An argmax one-hot function for arbitrary shapes."""
  inputs_flat = jnp.reshape(inputs, (-1))
  flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
  return jnp.reshape(flat_one_hot, inputs.shape)


def argmax_tree(x):
  return jax.tree.map(one_hot_argmax, x)


class MakePertTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.RandomState(0)

    self.rng_jax = jax.random.key(0)
    # Monte-Carlo estimate precision is const/sqrt(num_samples).
    # To avoid false positives, we wish to check approximately two decimal
    # places, so pick num_samples more than 10^4
    self.num_samples = 40_000
    # check a logrithmic range of sigma values, smaller values of sigma give
    # larger derivatives, and require more samples to estimate correctly.
    # Conversely, larger values of sigma increase the variance of the function
    # we are estimating.
    self.sigmas = [0.5, 1.0, 5.0]

    self.tree_a = (rng.randn(20, 10), rng.randn(20))

    self.tree_a_dict_jax = (
        jnp.array((1.0, 4.0, 5.0)),
        {'k1': jnp.array((-1.0, 2.0))},
    )
    self.array_a = rng.randn(20)

    self.array_a_jax = jnp.array(self.array_a)
    self.array_small_jax = jnp.array([1.0, 2.0, 3.0, 4.0])

    weight_shapes = [(13, 4), (8, 13)]
    biases_shapes = [(13,), (8,)]

    example_tree = []

    for i in range(2):
      example_tree.append({
          'weights': jnp.ones(weight_shapes[i]),
          'biases': jnp.ones(biases_shapes[i]),
      })

    self.example_tree = example_tree
    self.element = jnp.array([1.0, 2.0, 3.0, 4.0])

    self.element_tree = [
        self.element,
        {'a': jnp.ones_like(self.element), 'b': jnp.zeros_like(self.element)},
    ]

    # linear function x -> Ax + b
    self.linear_A = jnp.array([[0.0, 1.0, 2.0],
                               [3.0, 4.0, 5.0]])
    self.linear_b = jnp.array([1.0,-2.0])
    self.linear_x_test = jnp.array([-1.0, 1.0, 0.0])

  def test_pert_argmax(self):
    """Test that make_perturbed_fun matches theoretical expression of gradient.

    Applies to the case of an argmax. Includes a test for the gradients and the
    Hessian of a scalar loss.
    """

    for x, sigma in itertools.product([self.array_a_jax, self.tree_a_dict_jax], self.sigmas):
      # pert_argmax_fun and its jacobian should be an unbiased estimator of
      # softmax_fun and its jacobian
      pert_argmax_fun = _make_pert.make_perturbed_fun(
          argmax_tree, self.num_samples, sigma
      )
      softmax_fun = lambda t, sigma = sigma: jax.tree.map(lambda x: jax.nn.softmax(x / sigma), t)

      expected = softmax_fun(x)
      got = pert_argmax_fun(x,self.rng_jax)
      chex.assert_trees_all_equal_shapes(got, expected)
      chex.assert_trees_all_close(got, expected, atol=2e-1)

      expected = jax.jacobian(softmax_fun)(x)
      got = jax.jacobian(pert_argmax_fun)(x,self.rng_jax)
      chex.assert_trees_all_equal_shapes(got, expected)
      chex.assert_trees_all_close(got, expected, atol=2e-1)

      def pert_loss(inputs, rng, pert_argmax_fun=pert_argmax_fun):
        pert_argmax = pert_argmax_fun(inputs, rng)
        return optax.tree.sum(jax.tree.map(lambda x: x**2 + jnp.cos(x), pert_argmax))

      def exact_loss(inputs, softmax_fun=softmax_fun):
        softmax = softmax_fun(inputs)
        return optax.tree.sum(jax.tree.map(lambda x: x**2 + jnp.cos(x), softmax))

      expected = jax.grad(exact_loss)(x)
      got = jax.grad(pert_loss)(x,self.rng_jax)
      chex.assert_trees_all_equal_shapes(got, expected)
      chex.assert_trees_all_close(got, expected, atol=2e-1)

      # Do not check the hessian, as the second order derivative is currently not
      # implemented with Monte-Carlo methods, and there is no reason to expect correct
      # values

  def test_values_on_tree(self):
    """Test that the perturbations are well applied for functions on trees.

    Checks that small perturbations on the inputs have small effects on values
    """

    def apply_both(tree, element):
      x = element
      for i in range(2):
        s_tree = tree[i]
        x = jnp.dot(s_tree['weights'], x) + s_tree['biases']
      return x

    def apply_element_tree(tree):
      leaves, _ = jax.tree.flatten(tree)
      return_tree = jax.tree.map(partial(apply_both, tree), self.element_tree)
      return_tree.append(sum(jnp.sum(leaf) for leaf in leaves))
      return return_tree

    tree_out = apply_element_tree(self.example_tree)
    tree_noise = optax.tree.random_like(
        self.rng_jax, self.example_tree, sampler=_make_pert.Normal().sample
    )
    tree_out_noisy = apply_element_tree(
        optax.tree.add_scale(self.example_tree, 1e-4, tree_noise)
    )
    chex.assert_trees_all_close(tree_out, tree_out_noisy, rtol=1e-4)

    def loss(tree):
      pred = apply_element_tree(tree)
      pred_true = apply_element_tree(self.example_tree)
      tree_loss = jax.tree.map(lambda x, y: (x - y) ** 2, pred, pred_true)
      list_loss = jax.tree.reduce(operator.add, tree_loss)
      return jax.tree.map(lambda *leaves: sum(leaves) / len(leaves), list_loss)

    loss_pert = _make_pert.make_perturbed_fun(
        loss, num_samples=100, sigma=0.1, noise=_make_pert.Normal()
    )
    rngs = jax.random.split(self.rng_jax, 3)
    low_loss = loss_pert(self.example_tree, rngs[0])
    high_loss = loss_pert(
        optax.tree.random_like(rngs[1], self.example_tree), rngs[1]
    )
    np.testing.assert_array_less(low_loss, high_loss)


  def test_pert_linear_function(self):
    """Test that the peturbed function and its derivative is accurate in the
    transparent case of a linear function.
    """

    for sigma in self.sigmas:
      # pert_linear_fun and its jacobian should be an unbiased estimator of
      # linear_fun and its jacobian
      linear_fun = lambda x: self.linear_A @ x + self.linear_b
      pert_linear_fun = _make_pert.make_perturbed_fun(
              linear_fun, self.num_samples, sigma, _make_pert.Normal()
          )
      x = self.linear_x_test

      expected = linear_fun(x)
      got = pert_linear_fun(x,self.rng_jax)
      chex.assert_trees_all_equal_shapes(got, expected)
      chex.assert_trees_all_close(got, expected, atol=2e-1 * sigma)

      expected = jax.jacobian(linear_fun)(x)
      got = jax.jacobian(pert_linear_fun)(x,self.rng_jax)
      chex.assert_trees_all_equal_shapes(got, expected)
      chex.assert_trees_all_close(got, expected, atol=2e-1)

  def test_nonflat_shape(self):
    """Test that https://github.com/google-deepmind/optax/issues/1309 is fixed
    """
    x = jax.numpy.ones((1,2))
    f = lambda x: x[0,0]
    fp = _make_pert.make_perturbed_fun(f, 10)
    jax.grad(fp)(x, self.rng_jax)


if __name__ == '__main__':
  absltest.main()
