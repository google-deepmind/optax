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
"""A basic MNIST example using the Adam optimizer and lookahead wrapper."""
import functools
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple

from absl import app
import haiku as hk
import jax
from jax import random
import jax.numpy as jnp
import optax

# pylint: disable=g-bad-import-order
import datasets  # Located in the examples folder.
# pylint: enable=g-bad-import-order

LEARNING_RATE = 0.002
SLOW_LEARNING_RATE = 0.5
SYNC_PERIOD = 5
HIDDEN_DIMS = [1000, 1000]
BATCH_SIZE = 128
N_EPOCHS = 5
SEED = 1


def categorical_crossentropy(
    logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  losses = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=1)
  return jnp.mean(losses)


def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  predictions = jnp.argmax(logits, axis=-1)
  return jnp.mean(jnp.argmax(labels, axis=-1) == predictions)


def model_accuracy(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    batch_iterable: Iterable[Mapping[str, jnp.ndarray]]) -> jnp.ndarray:
  """Returns the accuracy of a model on a batched dataset."""
  accuracy_sum = dataset_size = 0
  for batch in batch_iterable:
    # Take batch size into account in case there is a smaller remainder batch.
    batch_size = batch['image'].shape[0]
    logits = model(batch['image'])
    accuracy_sum += accuracy(logits, batch['label']) * batch_size
    dataset_size += batch_size

  return accuracy_sum / dataset_size


# Optax is agnostic to which (if any) neural network library is used. Below we
# provide a Haiku version.
def _make_model(
    layer_dims: Sequence[int]) -> Tuple[Callable[..., Any], Callable[..., Any]]:
  """Simple multi-layer perceptron model for image classification."""

  @hk.transform
  def mlp_model(inputs: jnp.ndarray) -> jnp.ndarray:
    flattened = hk.Flatten()(inputs)
    return hk.nets.MLP(layer_dims)(flattened)

  return hk.without_apply_rng(mlp_model)


def main(unused_argv) -> None:
  train_dataset = datasets.load_image_dataset('mnist', BATCH_SIZE)
  test_dataset = datasets.load_image_dataset(
      'mnist', BATCH_SIZE, datasets.Split.TEST)
  num_classes = train_dataset.element_spec['label'].shape[1]

  init_params_fn, apply_params_fn = _make_model((*HIDDEN_DIMS, num_classes))

  # Set up the fast optimizer (adam) and wrap lookahead around it.
  fast_optimizer = optax.adam(LEARNING_RATE)
  optimizer = optax.lookahead(
      fast_optimizer, SYNC_PERIOD, SLOW_LEARNING_RATE)

  def get_loss(fast_params, batch):
    logits = apply_params_fn(fast_params, batch['image'])
    return categorical_crossentropy(logits, batch['label'])

  @jax.jit
  def train_step(params, optimizer_state, batch):
    grads = jax.grad(get_loss)(params.fast, batch)
    updates, opt_state = optimizer.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), opt_state

  example_input = next(train_dataset.as_numpy_iterator())['image']
  initial_params = init_params_fn(random.PRNGKey(SEED), example_input)

  # The lookahead optimizer wrapper keeps a pair of slow and fast parameters. To
  # initialize them, we create a pair of synchronized parameters from the
  # initial model parameters. The first line below is only necessary for the
  # lookahead wrapper; without it the initial parameters could be used in the
  # initialization function of the optimizer directly.
  params = optax.LookaheadParams.init_synced(initial_params)
  opt_state = optimizer.init(params)

  # Training loop
  for epoch in range(N_EPOCHS):
    for batch in train_dataset.as_numpy_iterator():
      params, opt_state = train_step(params, opt_state, batch)

    # Validation is done on the slow lookahead parameters.
    eval_model = functools.partial(apply_params_fn, params.slow)
    test_acc = model_accuracy(eval_model, test_dataset.as_numpy_iterator())
    print(f'Epoch {epoch+1}: test acc: {test_acc:.2f}')


if __name__ == '__main__':
  app.run(main)
