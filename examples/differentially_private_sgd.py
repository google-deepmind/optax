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
r"""Trains a differentially private convolutional neural network on MNIST.

A large portion of this code is forked from the differentially private SGD
example in the JAX repo:
https://github.com/google/jax/blob/master/examples/differentially_private_sgd.py

Differentially Private Stochastic Gradient Descent
(https://arxiv.org/abs/1607.00133) requires clipping the per-example parameter
gradients, which is non-trivial to implement efficiently for convolutional
neural networks.  The JAX XLA compiler shines in this setting by optimizing the
minibatch-vectorized computation for convolutional architectures. Train time
takes a few seconds per epoch on a commodity GPU.

The results match those in the reference TensorFlow baseline implementation:
https://github.com/tensorflow/privacy/tree/master/tutorials

Example invocations from within the `examples/` directory:

  # this non-private baseline should get ~99% acc
  python differentially_private_sgd.py \
    --dpsgd=False \
    --learning_rate=.1 \
    --epochs=20

  # this private baseline should get ~95% acc
  python differentially_private_sgd.py \
   --dpsgd=True \
   --noise_multiplier=1.3 \
   --l2_norm_clip=1.5 \
   --epochs=15 \
   --learning_rate=.25

  # this private baseline should get ~96.6% acc
  python differentially_private_sgd.py \
   --dpsgd=True \
   --noise_multiplier=1.1 \
   --l2_norm_clip=1.0 \
   --epochs=60 \
   --learning_rate=.15

  # this private baseline should get ~97% acc
  python differentially_private_sgd.py \
   --dpsgd=True \
   --noise_multiplier=0.7 \
   --l2_norm_clip=1.5 \
   --epochs=45 \
   --learning_rate=.25
"""

import time
import warnings

from absl import app
from absl import flags
import dp_accounting
import jax
from jax.example_libraries import stax
import jax.numpy as jnp
import optax

# pylint: disable=g-bad-import-order
import datasets  # Located in the examples folder.
# pylint: enable=g-bad-import-order

NUM_EXAMPLES = 60_000
FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('seed', 1337, 'Seed for JAX PRNG')
flags.DEFINE_float('delta', 1e-5, 'Target delta used to compute privacy spent.')


init_random_params, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(10),
)


def compute_epsilon(steps, target_delta=1e-5):
  if NUM_EXAMPLES * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = FLAGS.batch_size / float(NUM_EXAMPLES)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  accountant = dp_accounting.rdp.RdpAccountant(orders)
  accountant.compose(dp_accounting.PoissonSampledDpEvent(
      q, dp_accounting.GaussianDpEvent(FLAGS.noise_multiplier)), steps)
  return accountant.get_epsilon(target_delta)


def loss_fn(params, batch):
  logits = predict(params, batch['image'])
  return optax.softmax_cross_entropy(logits, batch['label']).mean(), logits


@jax.jit
def test_step(params, batch):
  loss, logits = loss_fn(params, batch)
  accuracy = (logits.argmax(1) == batch['label'].argmax(1)).mean()
  return loss, accuracy * 100


def main(_):
  train_dataset = datasets.load_image_dataset('mnist', FLAGS.batch_size)
  test_dataset = datasets.load_image_dataset('mnist', NUM_EXAMPLES,
                                             datasets.Split.TEST)
  full_test_batch = next(test_dataset.as_numpy_iterator())

  if FLAGS.dpsgd:
    tx = optax.dpsgd(learning_rate=FLAGS.learning_rate,
                     l2_norm_clip=FLAGS.l2_norm_clip,
                     noise_multiplier=FLAGS.noise_multiplier,
                     seed=FLAGS.seed)
  else:
    tx = optax.sgd(learning_rate=FLAGS.learning_rate)

  @jax.jit
  def train_step(params, opt_state, batch):
    grad_fn = jax.grad(loss_fn, has_aux=True)
    if FLAGS.dpsgd:
      # Insert dummy dimension in axis 1 to use jax.vmap over the batch
      batch = jax.tree_util.tree_map(lambda x: x[:, None], batch)
      # Use jax.vmap across the batch to extract per-example gradients
      grad_fn = jax.vmap(grad_fn, in_axes=(None, 0))

    grads, _ = grad_fn(params, batch)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  key = jax.random.PRNGKey(FLAGS.seed)
  _, params = init_random_params(key, (-1, 28, 28, 1))
  opt_state = tx.init(params)

  print('\nStarting training...')
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    for batch in train_dataset.as_numpy_iterator():
      params, opt_state = train_step(params, opt_state, batch)
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} in {epoch_time:0.2f} seconds.')

    # Evaluate test accuracy
    test_loss, test_acc = test_step(params, full_test_batch)
    print(f'Test Loss: {test_loss:.2f}  Test Accuracy (%): {test_acc:.2f}).')

    # Determine privacy loss so far
    if FLAGS.dpsgd:
      steps = epoch * NUM_EXAMPLES // FLAGS.batch_size
      eps = compute_epsilon(steps, FLAGS.delta)
      print(f'For delta={FLAGS.delta:.0e}, the current epsilon is: {eps:.2f}.')
    else:
      print('Trained with vanilla non-private SGD optimizer.')


if __name__ == '__main__':
  app.run(main)
