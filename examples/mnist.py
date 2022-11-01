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
"""An example showing how to train an MLP classifier on MNIST using optax."""
import functools
from typing import Callable, Iterable, Mapping, Sequence

from absl import app
import chex
import haiku as hk
import jax
from jax import random
import jax.numpy as jnp
import optax

# pylint: disable=g-bad-import-order
import datasets  # Located in the examples folder.
# pylint: enable=g-bad-import-order

LEARNING_RATE = 0.002
DEFAULT_HIDDEN_SIZES = (1000, 1000)
BATCH_SIZE = 128
N_EPOCHS = 5
SEED = 1


@jax.jit
def _single_batch_accuracy(logits: chex.Array,
                           labels: chex.Array) -> chex.Array:
  """Returns the accuracy for a batch of logits and labels."""
  predictions = jnp.argmax(logits, axis=-1)
  return jnp.mean(jnp.argmax(labels, axis=-1) == predictions)


def model_accuracy(model: Callable[[chex.Array], chex.Array],
                   dataset: Iterable[Mapping[str, chex.Array]]) -> chex.Array:
  """Returns the accuracy of a model on a batched dataset."""
  accuracy_sum = dataset_size = 0
  for batch in dataset:
    # Take batch size into account in case there is a smaller remainder batch.
    batch_size = batch['image'].shape[0]
    logits = model(batch['image'])
    accuracy_sum += _single_batch_accuracy(logits, batch['label']) * batch_size
    dataset_size += batch_size

  return accuracy_sum / dataset_size


# Optax is agnostic to which (if any) neural network library is used. Below we
# provide a Haiku version.
def build_model(layer_dims: Sequence[int]) -> hk.Transformed:
  """Simple multi-layer perceptron model for image classification."""

  @hk.transform
  def mlp_model(inputs: chex.Array) -> chex.Array:
    flattened = hk.Flatten()(inputs)
    return hk.nets.MLP(layer_dims)(flattened)

  return hk.without_apply_rng(mlp_model)


def train_on_mnist(optimizer: optax.GradientTransformation,
                   hidden_sizes: Sequence[int]) -> float:
  """Trains an MLP on MNIST using a given optimizer.

  Args:
    optimizer: Optax optimizer to use for training.
    hidden_sizes: Hidden layer sizes of the MLP.

  Returns:
    The final test accuracy.
  """
  train_dataset = datasets.load_image_dataset('mnist', BATCH_SIZE)
  test_dataset = datasets.load_image_dataset('mnist', BATCH_SIZE,
                                             datasets.Split.TEST)
  num_classes = train_dataset.element_spec['label'].shape[1]

  init_params_fn, apply_params_fn = build_model((*hidden_sizes, num_classes))

  def get_loss(params, batch):
    logits = apply_params_fn(params, batch['image'])
    return jnp.mean(optax.softmax_cross_entropy(logits, batch['label']))

  @jax.jit
  def train_step(params, optimizer_state, batch):
    grads = jax.grad(get_loss)(params, batch)
    updates, opt_state = optimizer.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), opt_state

  example_input = next(train_dataset.as_numpy_iterator())['image']
  params = init_params_fn(random.PRNGKey(SEED), example_input)
  opt_state = optimizer.init(params)

  # Training loop
  for epoch in range(N_EPOCHS):
    for batch in train_dataset.as_numpy_iterator():
      params, opt_state = train_step(params, opt_state, batch)

    eval_model = functools.partial(apply_params_fn, params)
    test_acc = model_accuracy(eval_model, test_dataset.as_numpy_iterator())
    print(f'Epoch {epoch+1}: test acc: {test_acc:.2f}')

  return test_acc


def main(unused_argv):
  """Trains an MLP on MNIST using the adam optimizers."""
  return train_on_mnist(optax.adam(LEARNING_RATE), DEFAULT_HIDDEN_SIZES)


if __name__ == '__main__':
  app.run(main)
