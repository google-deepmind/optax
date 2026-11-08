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

"""Tests for methods in `optax.projections.py`."""

from functools import partial  # pylint: disable=g-importing-member

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax import projections as proj
from optax._src import test_utils
import optax.tree


def projection_simplex_jacobian(projection):
  """Theoretical expression for the Jacobian of projection_simplex."""
  support = (projection > 0).astype(jnp.int32)
  cardinality = jnp.count_nonzero(support)
  return jnp.diag(support) - jnp.outer(support, support) / cardinality


def _derivative_tree(values):
  # Unequal leaf ranks and a None node catch lost cross-leaf dependencies.
  return {'a': values[:2], 'b': (values[2], None)}


def _derivative_array(tree):
  return jnp.concatenate((tree['a'], tree['b'][0][None]))


def _check_projection_derivatives(function, x, scale, jacobian, scale_gradient,
                                  use_jit, forward):
  def flat_function(values, radius):
    return _derivative_array(function(_derivative_tree(values), radius))

  differentiate = jax.jacfwd if forward else jax.jacrev
  derivative = differentiate(flat_function, argnums=(0, 1))
  if use_jit:
    derivative = jax.jit(derivative)
  actual_x, actual_scale = derivative(jnp.asarray(x), jnp.asarray(scale))
  np.testing.assert_allclose(actual_x, jacobian, atol=2e-6, rtol=2e-6)
  np.testing.assert_allclose(
      actual_scale, scale_gradient, atol=2e-6, rtol=2e-6
  )


class ProjectionsTest(parameterized.TestCase):

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_hyperplane_analytical_derivatives(self, use_jit, forward):
    a = np.array([1., 2., -2.])
    _check_projection_derivatives(
        lambda tree, b: proj.projection_hyperplane(
            tree, _derivative_tree(a), b
        ),
        [2., 1., -1.], 1.0, np.eye(3) - np.outer(a, a) / 9., a / 9.,
        use_jit, forward,
    )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_vector_analytical_derivatives(self, use_jit, forward):
    a = np.array([1., 2., -2.])
    _check_projection_derivatives(
        lambda tree, _: proj.projection_vector(tree, _derivative_tree(a)),
        [2., 1., -1.], 1.0, np.outer(a, a) / 9., np.zeros(3),
        use_jit, forward,
    )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_linf_ball_analytical_derivatives(self, use_jit, forward):
    _check_projection_derivatives(
        proj.projection_linf_ball, [-2.0, 0.4, 3.0], 1.0,
        np.diag([0., 1., 0.]), np.array([-1., 0., 1.]), use_jit, forward,
    )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_l2_ball_analytical_derivatives(self, use_jit, forward):
    x = np.array([1., -2., 2.])
    for radius in (2., 4.):
      with self.subTest(radius=radius):
        jacobian = (radius / 3.) * (np.eye(3) - np.outer(x, x) / 9.)
        radius_gradient = x / 3.
        if radius > 3.:
          jacobian, radius_gradient = np.eye(3), np.zeros(3)
        _check_projection_derivatives(
            proj.projection_l2_ball, x, radius, jacobian, radius_gradient,
            use_jit, forward,
        )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_l2_sphere_analytical_derivatives(self, use_jit, forward):
    x = np.array([1., -2., 2.])
    # Normalization removes the radial component of each input tangent.
    jacobian = (2. / 3.) * (np.eye(3) - np.outer(x, x) / 9.)
    _check_projection_derivatives(
        proj.projection_l2_sphere, x, 2.0, jacobian, x / 3.,
        use_jit, forward,
    )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_l1_ball_analytical_derivatives(self, use_jit, forward):
    for x, jacobian, radius_gradient in (
        ([0.1, -0.2, 0.3], np.eye(3), np.zeros(3)),
        ([1.5, -1.2, 0.1],
         np.array([[0.5, 0.5, 0.], [0.5, 0.5, 0.], [0., 0., 0.]]),
         np.array([0.5, -0.5, 0.])),
    ):
      with self.subTest(x=x):
        _check_projection_derivatives(
            proj.projection_l1_ball, x, 2.0, jacobian, radius_gradient,
            use_jit, forward,
        )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_l1_sphere_analytical_derivatives(self, use_jit, forward):
    # Mixed signs require sign changes in the off-diagonal Jacobian entries.
    _check_projection_derivatives(
        proj.projection_l1_sphere, [1.5, -1.2, 0.1], 2.0,
        np.array([[0.5, 0.5, 0.], [0.5, 0.5, 0.], [0., 0., 0.]]),
        np.array([0.5, -0.5, 0.]), use_jit, forward,
    )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_simplex_analytical_derivatives(self, use_jit, forward):
    # The first two coordinates remain in the support; the third is inactive.
    _check_projection_derivatives(
        proj.projection_simplex, [1.5, 1.2, -0.5], 2.0,
        np.array([[0.5, -0.5, 0.], [-0.5, 0.5, 0.], [0., 0., 0.]]),
        np.array([0.5, 0.5, 0.]), use_jit, forward,
    )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_hypercube_analytical_derivatives(self, use_jit, forward):
    x = jnp.array([-0.5, 0.4, 2.0])
    differentiate = jax.jacfwd if forward else jax.jacrev
    derivative = differentiate(proj.projection_hypercube, argnums=(0, 1))
    if use_jit:
      derivative = jax.jit(derivative)
    actual_x, actual_scale = derivative(x, jnp.asarray(1.0))
    np.testing.assert_allclose(actual_x, np.diag([0., 1., 0.]), atol=1e-6)
    np.testing.assert_allclose(actual_scale, [0., 0., 1.], atol=1e-6)

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_box_analytical_derivatives(self, use_jit, forward):
    # Vary both bounds together; only clipped coordinates follow the bounds.
    def project(tree, shift):
      lower = _derivative_tree(jnp.array([-1., -1., -1.]) + shift)
      upper = _derivative_tree(jnp.array([1., 1., 1.]) + shift)
      return proj.projection_box(tree, lower, upper)

    _check_projection_derivatives(
        project, [-2.0, 0.4, 3.0], 0.0,
        np.diag([0., 1., 0.]), np.array([1., 0., 1.]), use_jit, forward,
    )

  @parameterized.product(use_jit=(False, True), forward=(False, True))
  def test_non_negative_analytical_derivatives(self, use_jit, forward):
    _check_projection_derivatives(
        lambda tree, _: proj.projection_non_negative(tree),
        [-2.0, 0.4, 3.0], 1.0, np.diag([0., 1., 1.]), np.zeros(3),
        use_jit, forward,
    )

  def setUp(self):
    super().setUp()
    array_1d = jnp.array([0.5, 2.1, -3.5])
    array_2d = jnp.array([[0.5, 2.1, -3.5], [1.0, 2.0, 3.0]])
    tree = (array_1d, array_1d)
    self.data = {
        'array_1d': array_1d,
        'array_2d': array_2d,
        'tree': tree,
    }
    self.fns = {
        'l1': (proj.projection_l1_ball, partial(optax.tree.norm, ord=1)),
        'l2': (proj.projection_l2_ball, optax.tree.norm),
        'linf': (proj.projection_linf_ball,
                 partial(optax.tree.norm, ord='inf')),
    }

  def test_projection_non_negative(self):
    with self.subTest('with an array'):
      x = jnp.array([-1.0, 2.0, 3.0])
      expected = jnp.array([0, 2.0, 3.0])
      np.testing.assert_array_equal(proj.projection_non_negative(x), expected)

    with self.subTest('with a tuple'):
      np.testing.assert_array_equal(
          proj.projection_non_negative((x, x)), (expected, expected)
      )

    with self.subTest('with nested pytree'):
      tree_x = (-1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
      tree_expected = (0.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
      test_utils.assert_trees_all_equal(
          proj.projection_non_negative(tree_x), tree_expected
      )

  def test_projection_box(self):
    with self.subTest('lower and upper are scalars'):
      lower, upper = 0.0, 2.0
      x = jnp.array([-1.0, 2.0, 3.0])
      expected = jnp.array([0, 2.0, 2.0])
      np.testing.assert_array_equal(
          proj.projection_box(x, lower, upper), expected
      )

    with self.subTest('lower and upper values are arrays'):
      lower_arr = jnp.ones(len(x)) * lower
      upper_arr = jnp.ones(len(x)) * upper
      np.testing.assert_array_equal(
          proj.projection_box(x, lower_arr, upper_arr), expected
      )

    with self.subTest('lower and upper are tuples of arrays'):
      lower_tuple = (lower, lower)
      upper_tuple = (upper, upper)
      test_utils.assert_trees_all_equal(
          proj.projection_box((x, x), lower_tuple, upper_tuple),
          (expected, expected),
      )

    with self.subTest('lower and upper are pytrees'):
      tree = (-1.0, {'k1': 2.0, 'k2': (2.0, 3.0)}, 3.0)
      expected = (0.0, {'k1': 2.0, 'k2': (2.0, 2.0)}, 2.0)
      lower_tree = (0.0, {'k1': 0.0, 'k2': (0.0, 0.0)}, 0.0)
      upper_tree = (2.0, {'k1': 2.0, 'k2': (2.0, 2.0)}, 2.0)
      test_utils.assert_trees_all_equal(
          proj.projection_box(tree, lower_tree, upper_tree), expected
      )

  def test_projection_hypercube(self):
    x = jnp.array([-1.0, 2.0, 0.5])

    with self.subTest('with default scale'):
      expected = jnp.array([0, 1.0, 0.5])
      np.testing.assert_array_equal(proj.projection_hypercube(x), expected)

    with self.subTest('with scalar scale'):
      expected = jnp.array([0, 0.8, 0.5])
      np.testing.assert_array_equal(proj.projection_hypercube(x, 0.8), expected)

    with self.subTest('with array scales'):
      scales = jnp.ones(len(x)) * 0.8
      np.testing.assert_array_equal(
          proj.projection_hypercube(x, scales), expected
      )

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_array(self, scale):
    rng = np.random.RandomState(0)
    x = rng.randn(50).astype(np.float32)
    p = proj.projection_simplex(x, scale)

    np.testing.assert_almost_equal(jnp.sum(p), scale, decimal=4)
    self.assertTrue(jnp.all(0 <= p))
    self.assertTrue(jnp.all(p <= scale))

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_pytree(self, scale):
    pytree = {'w': jnp.array([2.5, 3.2]), 'b': 0.5}
    new_pytree = proj.projection_simplex(pytree, scale)
    np.testing.assert_almost_equal(optax.tree.sum(new_pytree), scale, decimal=4)

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_edge_case(self, scale):
    p = proj.projection_simplex(jnp.array([0.0, 0.0, -jnp.inf]), scale)
    np.testing.assert_array_almost_equal(
        p, jnp.array([scale / 2, scale / 2, 0.0])
    )

  def test_projection_simplex_jacobian(self):
    rng = np.random.RandomState(0)

    x = rng.rand(5).astype(np.float32)
    v = rng.randn(5).astype(np.float32)

    jac_rev = jax.jacrev(proj.projection_simplex)(x)
    jac_fwd = jax.jacfwd(proj.projection_simplex)(x)

    with self.subTest('Check against theoretical expression'):
      p = proj.projection_simplex(x)
      jac_true = projection_simplex_jacobian(p)

      np.testing.assert_array_almost_equal(jac_true, jac_fwd)
      np.testing.assert_array_almost_equal(jac_true, jac_rev)

    with self.subTest('Check against finite difference'):
      jvp = jax.jvp(proj.projection_simplex, (x,), (v,))[1]
      eps = 1e-4
      jvp_finite_diff = (
          proj.projection_simplex(x + eps * v)
          - proj.projection_simplex(x - eps * v)
      ) / (2 * eps)
      np.testing.assert_array_almost_equal(jvp, jvp_finite_diff, decimal=3)

    with self.subTest('Check vector-Jacobian product'):
      (vjp,) = jax.vjp(proj.projection_simplex, x)[1](v)
      np.testing.assert_array_almost_equal(vjp, jnp.dot(v, jac_true))

    with self.subTest('Check Jacobian-vector product'):
      jvp = jax.jvp(proj.projection_simplex, (x,), (v,))[1]
      np.testing.assert_array_almost_equal(jvp, jnp.dot(jac_true, v))

  @parameterized.parameters(1.0, 0.8)
  def test_projection_simplex_vmap(self, scale):
    rng = np.random.RandomState(0)
    x = rng.randn(3, 50).astype(np.float32)
    scales = jnp.full(len(x), scale)

    p = jax.vmap(proj.projection_simplex)(x, scales)
    np.testing.assert_array_almost_equal(jnp.sum(p, axis=1), scales)
    np.testing.assert_array_equal(True, 0 <= p)
    np.testing.assert_array_equal(True, p <= scale)

  @parameterized.product(
      data_key=['array_1d', 'array_2d', 'tree'], scale=[1.0, 3.21]
  )
  def test_projection_l1_sphere(self, data_key, scale):
    x = self.data[data_key]
    p = proj.projection_l1_sphere(x, scale)
    np.testing.assert_almost_equal(optax.tree.norm(p, ord=1), scale, decimal=4)

  @parameterized.product(
      data_key=['array_1d', 'array_2d', 'tree'], scale=[1.0, 3.21]
  )
  def test_projection_l2_sphere(self, data_key, scale):
    x = self.data[data_key]
    p = proj.projection_l2_sphere(x, scale)
    np.testing.assert_almost_equal(optax.tree.norm(p), scale, decimal=4)

  @parameterized.product(
      data_key=['array_1d', 'array_2d', 'tree'],
      norm=['l1', 'l2', 'linf'],
  )
  def test_projection_ball(self, data_key, norm):
    """Check correctness of the projection onto a ball."""
    proj_fun, norm_fun = self.fns[norm]
    x = self.data[data_key]

    norm_value = norm_fun(x)

    with self.subTest('Check when input is already in the ball'):
      big_radius = norm_value * 2
      p = proj_fun(x, big_radius)
      np.testing.assert_array_almost_equal(x, p)

    with self.subTest('Check when input is on the boundary of the ball'):
      p = proj_fun(x, norm_value)
      np.testing.assert_array_almost_equal(x, p)

    with self.subTest('Check when input is outside the ball'):
      small_radius = norm_value / 2
      p = proj_fun(x, small_radius)
      np.testing.assert_almost_equal(norm_fun(p), small_radius, decimal=4)

  def test_projection_l2_ball_grad_at_zero(self):
    grad = jax.grad(proj.projection_l2_ball)(0.0)
    assert not jnp.isnan(grad)
    assert grad == 1.0

  def test_projection_l1_ball_grad_at_zero(self):
    grad = jax.grad(proj.projection_l1_ball)(0.0)
    assert not jnp.isnan(grad)
    assert grad == 1.0

  def test_projection_vector(self):
    x = (1.0, 2.0)
    a = (3.0, 4.0)
    y_actual = proj.projection_vector(x, a)
    y_expected = (33 / 25, 44 / 25)
    assert optax.tree.allclose(y_actual, y_expected)

  def test_projection_hyperplane(self):
    x = (1.0, 2.0)
    a = (3.0, 4.0)
    b = 5.0
    y_actual = proj.projection_hyperplane(x, a, b)
    y_expected = (7 / 25, 26 / 25)
    assert optax.tree.allclose(y_actual, y_expected)

  def test_projection_halfspace_1(self):
    x = (1.0, 2.0)
    a = (3.0, 4.0)
    b = 5.0
    y_actual = proj.projection_halfspace(x, a, b)
    y_expected = (7 / 25, 26 / 25)
    assert optax.tree.allclose(y_actual, y_expected)

  def test_projection_halfspace_2(self):
    x = (1.0, -2.0)
    a = (3.0, 4.0)
    b = 5.0
    y_actual = proj.projection_halfspace(x, a, b)
    y_expected = x
    assert optax.tree.allclose(y_actual, y_expected)


if __name__ == '__main__':
  absltest.main()
