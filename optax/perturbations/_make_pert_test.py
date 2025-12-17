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
import jax
import jax.numpy as jnp
import numpy as np

from optax._src import test_utils
from optax.perturbations import _make_pert
import optax.tree

# pylint: disable=g-doc-args


def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
  """An argmax one-hot function for arbitrary shapes."""
  inputs_flat = jnp.reshape(inputs, (-1))
  flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
  return jnp.reshape(flat_one_hot, inputs.shape)


def argmax_tree(x):
  return jax.tree.map(one_hot_argmax, x)


def simple_make_perturbed_fun(f, num_samples=1000, sigma=0.1,
                              noise=_make_pert.Gumbel()):
  # a simplified Monte Carlo estimate of E[f(x + σ Z)] where Z ~ noise
  # this simplified reference only works for differentiable f
  @jax.jit
  def g(key, x):
    zs_shape = optax.tree.batch_shape(x, (num_samples,))
    zs = optax.tree.random_like(key, zs_shape, noise.sample)
    xs = optax.tree.add_scale(x, sigma, zs)
    ys = jax.vmap(f)(xs)
    ys_mean = jax.tree.map(lambda leaf: jnp.mean(leaf, 0), ys)
    return ys_mean
  return g


# this tests includes long compilation time, so sharply limit the test cases
class MakePertTest(parameterized.TestCase):
  @parameterized.product(xtype=['tree', 'flat'], sigma=[0.5, 1.0])
  def test_pert_argmax(self, xtype, sigma):
    """Test that make_perturbed_fun matches theoretical expression of gradient.

    Applies to the case of an argmax. Includes a test for the jacobian and
    gradients.

    This test is probabilistic, and can fail with low probability. If this
    happens, try increasing num_samples or loosening the tolerances of the
    assertions.
    """

    num_samples = 1_000_000

    key = jax.random.key(0)
    if xtype == 'tree':
      key, key1, key2, key3, key4 = jax.random.split(key, 5)
      x = (jax.random.normal(key1, (4,)), {'k1': jax.random.normal(key2, (3,)),
                                           'k2': jax.random.normal(key3, (2,))},
           jax.random.normal(key4, (2,)))
    else:
      key, key1 = jax.random.split(key)
      x = jax.random.normal(key1, (20,))

    # pert_argmax_fun and its jacobian should be an unbiased estimator of
    # softmax_fun and its jacobian
    pert_argmax_fun = jax.jit(
        _make_pert.make_perturbed_fun(argmax_tree, num_samples, sigma)
    )
    softmax_fun = lambda x: jax.tree.map(lambda z: jax.nn.softmax(z / sigma), x)

    expected = softmax_fun(x)
    got = pert_argmax_fun(key, x)
    test_utils.assert_trees_all_equal_shapes(got, expected)
    test_utils.assert_trees_all_close(got, expected, atol=1e-1)

    expected = jax.jacobian(softmax_fun)(x)
    got = jax.jacobian(pert_argmax_fun, argnums=1)(key, x)
    test_utils.assert_trees_all_equal_shapes(got, expected)
    test_utils.assert_trees_all_close(got, expected, atol=1e-1)

    # test gradients for losses

    @jax.jit
    def pert_loss(key, inputs):
      pert_argmax = pert_argmax_fun(key, inputs)
      return optax.tree.sum(jax.tree.map(lambda x: x**2 + jnp.cos(x),
                                         pert_argmax))

    @jax.jit
    def exact_loss(inputs):
      softmax = softmax_fun(inputs)
      return optax.tree.sum(jax.tree.map(lambda x: x**2 + jnp.cos(x), softmax))

    expected = jax.grad(exact_loss)(x)
    got = jax.grad(pert_loss, argnums=1)(key, x)
    test_utils.assert_trees_all_equal_shapes(got, expected)
    test_utils.assert_trees_all_close(got, expected, atol=1e-1)

  def test_values_on_tree(self):
    """Test that the perturbations are well applied for functions on trees.

    Checks that small perturbations on the inputs have small effects on values
    """
    key = jax.random.key(0)

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

    @jax.jit
    def loss(tree):
      pred = apply_element_tree(tree)
      pred_true = apply_element_tree(example_tree)
      tree_loss = jax.tree.map(lambda x, y: (x - y) ** 2, pred, pred_true)
      list_loss = jax.tree.reduce(operator.add, tree_loss)
      return jax.tree.map(lambda *leaves: sum(leaves) / len(leaves), list_loss)

    loss_pert = jax.jit(_make_pert.make_perturbed_fun(
        loss, num_samples=100, sigma=0.1, noise=_make_pert.Normal()
    ))
    keys = jax.random.split(key, 3)
    low_loss = loss_pert(keys[0], example_tree)  # pytype: disable=wrong-arg-types # noqa: E501
    high_loss = loss_pert(keys[2],
                          optax.tree.random_like(keys[1], example_tree))
    np.testing.assert_array_less(low_loss, high_loss)

  # pylint: disable=invalid-name
  @parameterized.product(sigma=[0.5, 1.0])
  def test_pert_linear_function(self, sigma):
    """Ensure perturbed function and its derivative is accurate for a linear function.

    This test is probabilistic, and can fail with low probability. If this
    happens, try increasing num_samples or loosening the tolerances of the
    assertions.
    """  # noqa: E501

    num_samples = 1_000_000

    seed = 0
    m, n = (3, 2)
    key = jax.random.key(seed)
    key, key1, key2, key3 = jax.random.split(key, 4)
    A = jax.random.normal(key1, (m, n))  # pylint: disable=invalid-name
    b = jax.random.normal(key2, (m,))
    x = jax.random.normal(key3, (n,))

    # pert_linear_fun and its jacobian should be an unbiased estimator of
    # linear_fun and its jacobian
    linear_fun = lambda x: A @ x + b
    pert_linear_fun = jax.jit(_make_pert.make_perturbed_fun(
        linear_fun, num_samples, sigma, _make_pert.Normal()))

    expected = linear_fun(x)
    got = pert_linear_fun(key, x)
    test_utils.assert_trees_all_equal_shapes(got, expected)
    test_utils.assert_trees_all_close(got, expected, atol=1e-1)

    expected = jax.jacobian(linear_fun)(x)
    got = jax.jacobian(pert_linear_fun, argnums=1)(key, x)
    test_utils.assert_trees_all_equal_shapes(got, expected)
    test_utils.assert_trees_all_close(got, expected, atol=1e-1)
  # pylint: enable=invalid-name

  @parameterized.product(
      sigma=[0.5, 1.0], noise=[_make_pert.Gumbel(), _make_pert.Normal()]
  )
  def test_pert_differentiable(self, sigma, noise):
    """Ensure the perturbed function jacobian matches for a differentiable function.

    In the differentiable function case, test that the jacobian of the
    peturbed function agrees with the direct Monte-Carlo estimate over the
    derivative.

    This test is probabilistic, and can fail with low probability. If this
    happens, try increasing num_samples or loosening the tolerances of the
    assertions.
    """  # noqa: E501
    seed = 0
    num_samples = 1_000_000

    @jax.jit
    def f(x):
      # a differentiable test function
      x0, x1, x2 = x
      y0 = x0 * x1**2 * jnp.sin(x2 * x1 - x0) + jnp.cos(x0**3)
      y1 = x2 + x0 * jnp.sin(x1)
      return jnp.stack([y0, y1])

    f1 = jax.jit(_make_pert.make_perturbed_fun(
        f, num_samples=num_samples, sigma=sigma, noise=noise))
    f2 = simple_make_perturbed_fun(f, num_samples=num_samples, sigma=sigma,
                                   noise=noise)
    x = jnp.array([0.3, 0.4, 0.5])
    key = jax.random.key(seed)
    j1 = jax.jacobian(f1, argnums=1)(key, x)
    j2 = jax.jacobian(f2, argnums=1)(key, x)
    test_utils.assert_trees_all_close(j1, j2, atol=2e-1)

  def test_fun_derivative_not_used(self):
    # pylint: disable=line-too-long
    """Ensure the perturbed function derivative doesn't need the derivative of f."""  # noqa: E501
    # pylint: enable=line-too-long
    @jax.custom_jvp
    def f(x):
      return x

    @f.defjvp
    def _assert_unused(primal_in, tangent_in):
      del primal_in, tangent_in
      assert False

    key = jax.random.key(0)
    fp = _make_pert.make_perturbed_fun(f, 10)
    jax.grad(fp, argnums=1)(key, jnp.array(0.0))

  def test_nonflat_shape(self):
    """Ensure https://github.com/google-deepmind/optax/issues/1309 is fixed."""
    key = jax.random.key(0)
    x = jax.numpy.ones((1, 2))
    f = lambda x: x[0, 0]
    fp = _make_pert.make_perturbed_fun(f, 10)
    jax.grad(fp, argnums=1)(key, x)

  @parameterized.product(
      sigma=[0.5, 1.0], noise=[_make_pert.Gumbel(), _make_pert.Normal()]
  )
  def test_hessian(self, sigma, noise):
    """Test that hessian of perturbed function matches exact hessian."""
    fun = lambda x: 0.5*jnp.sum(x**2)
    fun_p = _make_pert.make_perturbed_fun(fun, 10**5, sigma, noise)
    x = jnp.array([0.0, 0.0])
    got = jax.hessian(fun_p, argnums=1)(jax.random.key(0), x)
    expected = jax.hessian(fun)(x)
    test_utils.assert_trees_all_close(got, expected, atol=1e-1)

if __name__ == '__main__':
  absltest.main()
