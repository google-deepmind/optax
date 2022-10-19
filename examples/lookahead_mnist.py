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
"""An MNIST example using the Adam optimizer and lookahead wrapper."""
import functools

from absl import app
import jax
from jax import random
import jax.numpy as jnp
import optax

# pylint: disable=g-bad-import-order
import datasets  # Located in the examples folder.
import mnist  # Located in the examples folder.
# pylint: enable=g-bad-import-order

LEARNING_RATE = 0.002
SLOW_LEARNING_RATE = 0.5
SYNC_PERIOD = 5
HIDDEN_SIZES = (1000, 1000)
BATCH_SIZE = 128
N_EPOCHS = 5
SEED = 1


def main(unused_argv) -> None:
  train_dataset = datasets.load_image_dataset('mnist', BATCH_SIZE)
  test_dataset = datasets.load_image_dataset('mnist', BATCH_SIZE,
                                             datasets.Split.TEST)
  num_classes = train_dataset.element_spec['label'].shape[1]

  init_params_fn, apply_params_fn = mnist.build_model(
      (*HIDDEN_SIZES, num_classes))

  # Set up the fast optimizer (adam) and wrap lookahead around it.
  fast_optimizer = optax.adam(LEARNING_RATE)
  optimizer = optax.lookahead(fast_optimizer, SYNC_PERIOD, SLOW_LEARNING_RATE)

  def get_loss(fast_params, batch):
    logits = apply_params_fn(fast_params, batch['image'])
    return jnp.mean(optax.softmax_cross_entropy(logits, batch['label']))

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
    test_acc = mnist.model_accuracy(eval_model,
                                    test_dataset.as_numpy_iterator())
    print(f'Epoch {epoch+1}: test acc: {test_acc:.2f}')

  return test_acc


if __name__ == '__main__':
  app.run(main)
