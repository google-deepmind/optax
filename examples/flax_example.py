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
"""A simple example of using Optax to train the parameters of a Flax module."""

from absl import app

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


def main(argv):
  del argv

  learning_rate = 1e-2
  n_training_steps = 100

  # Random number generator sequence.
  rng = jax.random.PRNGKey(0)
  rng1, rng2 = jax.random.split(rng)

  # Create a one linear layer instance.
  model = nn.Dense(features=5)

  # Initialise the parameters.
  params = model.init(rng2, jax.random.normal(rng1, (10,)))

  # Set problem dimensions.
  nsamples = 20
  xdim = 10
  ydim = 5

  # Generate random ground truth w and b.
  w = jax.random.normal(rng1, (xdim, ydim))
  b = jax.random.normal(rng2, (ydim,))

  # Generate samples with additional noise.
  ksample, knoise = jax.random.split(rng1)
  x_samples = jax.random.normal(ksample, (nsamples, xdim))
  y_samples = jnp.dot(x_samples, w) + b
  y_samples += 0.1 * jax.random.normal(knoise, (nsamples, ydim))

  # Define an MSE loss function.
  def make_mse_func(x_batched, y_batched):
    def mse(params):
      # Define the squared loss for a single (x, y) pair.
      def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y-pred, y-pred) / 2.0
      # Vectorise the squared error and compute the average of the loss.
      return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
    return jax.jit(mse)  # `jit` the result.

  # Instantiate the sampled loss.
  loss = make_mse_func(x_samples, y_samples)

  # Construct a simple Adam optimiser using the transforms in optax.
  # You could also just use the `optax.adam` alias, but we show here how
  # to do so manually so that you may construct your own `custom` optimiser.
  tx = optax.chain(
      # Set the parameters of Adam. Note the learning_rate is not here.
      optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
      # Put a minus sign to *minimise* the loss.
      optax.scale(-learning_rate)
  )

  # Create optimiser state.
  opt_state = tx.init(params)
  # Compute the gradient of the loss function.
  loss_grad_fn = jax.value_and_grad(loss)

  # Minimise the loss.
  for step in range(n_training_steps):
    # Compute gradient of the loss.
    loss_val, grads = loss_grad_fn(params)
    # Update the optimiser state, create an update to the params.
    updates, opt_state = tx.update(grads, opt_state)
    # Update the parameters.
    params = optax.apply_updates(params, updates)
    print(f'Loss[{step}] = {loss_val}')


if __name__ == '__main__':
  app.run(main)
