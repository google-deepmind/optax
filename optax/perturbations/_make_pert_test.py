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

"""Tests for optax.perturbations."""

import operator

from absl.testing import absltest
import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp
import numpy as np
from optax.perturbations import _make_pert
from optax.tree_utils import _tree_math as otu


def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
  """An argmax one-hot function for arbitrary shapes."""
  inputs_flat = jnp.reshape(inputs, (-1))
  flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
  return jnp.reshape(flat_one_hot, inputs.shape)

argmax_tree = lambda x: jtu.tree_map(one_hot_argmax, x)


class MakePertTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.RandomState(0)

    self.rng_jax = jax.random.PRNGKey(0)
    self.num_samples = 1_000
    self.sigma = 0.5

    self.tree_a = (rng.randn(20, 10), rng.randn(20))

    self.tree_a_dict_jax = (jnp.array((1.0, 4.0, 5.0)),
                            {'k1': jnp.array((1.0, 2.0)),
                             'k2': jnp.array((1.0, 1.0))},
                            jnp.array((1.0, 2.0)))
    self.array_a = rng.randn(20)

    self.array_a_jax = jnp.array(self.array_a)

    weight_shapes = [(13, 4), (8, 13)]
    biases_shapes = [(13,), (8,)]

    example_tree = []

    for i in range(2):
      example_tree.append(dict(weights=jnp.ones(weight_shapes[i]),
                               biases=jnp.ones(biases_shapes[i])))

    self.example_tree = example_tree
    self.element = jnp.array([1, 2, 3, 4])

    self.element_tree = [self.element,
                         {'a': jnp.ones_like(self.element),
                          'b': jnp.zeros_like(self.element)}]

  def test_pert_close_array(self):
    pert_argmax_fun = _make_pert.make_perturbed_fun(argmax_tree,
                                                    self.num_samples,
                                                    self.sigma)
    expected = pert_argmax_fun((self.array_a_jax), self.rng_jax)
    softmax_fun = lambda x: jax.nn.softmax(x / self.sigma)
    got = jtu.tree_map(softmax_fun, self.array_a_jax)
    np.testing.assert_array_almost_equal(expected, got, decimal=1)
    expected_dict = pert_argmax_fun((self.tree_a_dict_jax), rng=self.rng_jax)
    got_dict = jtu.tree_map(softmax_fun, self.tree_a_dict_jax)
    chex.assert_trees_all_close(expected_dict, got_dict, atol=2e-2)

  def test_values_and_grad(self):

    def apply_both(tree, element):
      x = element
      for i in range(2):
        s_tree = tree[i]
        x = jnp.dot(s_tree['weights'], x) + s_tree['biases']
      return x

    def apply_element_tree(tree):
      apply_tree = jtu.Partial(apply_both, tree)
      leaves, _ = jtu.tree_flatten(tree)
      return_tree = jtu.tree_map(apply_tree, self.element_tree)
      return_tree.append(sum([jnp.sum(leaf) for leaf in leaves]))
      return return_tree

    tree_out = apply_element_tree(self.example_tree)
    tree_noise = otu.tree_random_like(
        self.rng_jax, self.example_tree, sampler=_make_pert.Normal().sample
    )
    tree_out_noisy = apply_element_tree(
        otu.tree_add_scalar_mul(self.example_tree, 1e-4, tree_noise)
    )
    chex.assert_trees_all_close(tree_out, tree_out_noisy, rtol=1e-4)

    def loss(tree):
      pred = apply_element_tree(tree)
      pred_true = apply_element_tree(self.example_tree)
      tree_loss = jtu.tree_map(lambda x, y: (x - y) ** 2, pred, pred_true)
      list_loss = jtu.tree_reduce(operator.add, tree_loss)
      return _make_pert._tree_mean_across(list_loss)
    loss_pert = _make_pert.make_perturbed_fun(loss,
                                              num_samples=100,
                                              sigma=0.1,
                                              noise=_make_pert.Normal())
    rngs = jax.random.split(self.rng_jax, 3)
    low_loss = loss_pert(self.example_tree, rngs[0])
    high_loss = loss_pert(otu.tree_random_like(rngs[1], self.example_tree),
                          rngs[1])
    np.testing.assert_array_less(low_loss, high_loss)


if __name__ == '__main__':
  absltest.main()
