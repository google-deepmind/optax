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
"""A simple example of using Optax to train the parameters of a Haiku module."""

from absl import app

import haiku as hk
import jax
import jax.numpy as jnp
import optax


def main(argv):
  del argv

  learning_rate = 1e-2
  batch_size = 64
  input_size = 8
  n_training_steps = 100

  # Random number generator sequence.
  key_seq = hk.PRNGSequence(1729)

  # A simple Linear function.
  def forward_pass(x):
    return hk.Linear(10)(x)

  network = hk.without_apply_rng(hk.transform(forward_pass))

  # Some arbitrary loss.
  def mean_square_loss(params, x):
    output = network.apply(params, x)
    loss = jnp.sum(output**2)
    return loss

  # Construct a simple Adam optimiser using the transforms in optax.
  # You could also just use the `optax.adam` alias, but we show here how
  # to do so manually so that you may construct your own `custom` optimiser.
  opt_init, opt_update = optax.chain(
      # Set the parameters of Adam. Note the learning_rate is not here.
      optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
      # Put a minus sign to *minimise* the loss.
      optax.scale(-learning_rate)
  )

  # Initialise the model's parameters and the optimiser's state.
  # The `state` of an optimiser contains all statistics used by the
  # stateful transformations in the `chain` (in this case just `scale_by_adam`).
  params = network.init(next(key_seq), jnp.zeros([1, input_size]))
  opt_state = opt_init(params)

  # Minimise the loss.
  for step in range(n_training_steps):
    # Get input. Learn to minimize the input to 0.
    data = jax.random.normal(next(key_seq), [batch_size, input_size])
    # Compute gradient and loss.
    loss, grad = jax.value_and_grad(mean_square_loss)(params, data)
    print(f'Loss[{step}] = {loss}')
    # Transform the gradients using the optimiser.
    updates, opt_state = opt_update(grad, opt_state, params)
    # Update parameters.
    params = optax.apply_updates(params, updates)


if __name__ == '__main__':
  app.run(main)
