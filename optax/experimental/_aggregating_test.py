# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.random as jrd
from optax._src import alias
from optax._src import base
from optax._src import test_utils
from optax._src import update
from optax.experimental import _aggregating as aggregating


def _train(
    opt,
    accumulation_steps: int = 1,
    num_samples: int = 16,
    batch_size: int = 4,
    dim: int = 4,
    num_classes: int = 2,
    metrics_ema_decay: float = 0.0,
):
  """Synthetic training with the given optimizer."""
  microbatch_size = batch_size // accumulation_steps

  def data_iterator(key):
    inputs_key, targets_key = jrd.split(key)
    inputs = jrd.normal(inputs_key, (num_samples, dim))
    targets = jrd.normal(targets_key, (num_samples, num_classes))

    for i in range(0, num_samples, microbatch_size):
      yield inputs[i : i + microbatch_size], targets[i : i + microbatch_size]

  def loss_fun(params, batch):
    inputs, targets = batch
    return jnp.mean(jnp.sum((inputs.dot(params) - targets) ** 2, -1))

  data_key, param_key = jrd.split(jrd.key(0))
  full_data = [
      jnp.concatenate(a, axis=0) for a in zip(*data_iterator(data_key))
  ]
  params = jrd.normal(param_key, (dim, num_classes))

  @jax.jit
  def train_step(params, state, batch):
    mean_grads = None
    var_grads = None
    if isinstance(opt, aggregating.Aggregator):
      losses, grads = jax.vmap(jax.value_and_grad(loss_fun), (None, 0))(
          params, batch
      )
      loss = jnp.mean(losses)
      if accumulation_steps == 1:
        mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)
        var_grads = jax.tree.map(lambda g: jnp.var(g, axis=0, ddof=1), grads)
    else:
      loss, grads = jax.value_and_grad(loss_fun)(params, batch)
    updates, state = opt.update(grads, state)
    params = update.apply_updates(params, updates)
    return params, state, loss, mean_grads, var_grads

  state = opt.init(params)
  metrics = {}
  true_mean_grads_ema = jnp.zeros_like(params)
  true_var_grads_ema = jnp.zeros_like(params)
  for i, batch in enumerate(data_iterator(data_key)):
    full_batch_loss = loss_fun(params, full_data)
    params, state, loss, true_mean_grads, true_var_grads = train_step(
        params, state, batch
    )
    step_metrics = {'loss': loss, 'full_batch_loss': full_batch_loss}
    if isinstance(opt, aggregating.Aggregator) and (accumulation_steps == 1):
      true_mean_grads_ema, true_var_grads_ema = jax.tree.map(
          lambda x, y: (1.0 - metrics_ema_decay) * x + metrics_ema_decay * y,
          (true_mean_grads, true_var_grads),
          (true_mean_grads_ema, true_var_grads_ema),
      )
      unbiased_true_mean_grads_ema = true_mean_grads_ema / (
          1 - metrics_ema_decay ** (i + 1)
      )
      unbiased_true_var_grads_ema = true_var_grads_ema / (
          1 - metrics_ema_decay ** (i + 1)
      )
      step_metrics['true_mean_grads_ema'] = unbiased_true_mean_grads_ema
      step_metrics['true_var_grads_ema'] = unbiased_true_var_grads_ema
    try:
      mean_grads_ema, var_grads_ema = (
          aggregating.get_unbiased_mean_and_variance_ema(state)
      )
      step_metrics['mean_grads_ema'] = mean_grads_ema
      step_metrics['var_grads_ema'] = var_grads_ema
    except ValueError:
      pass
    if not metrics:
      for key in step_metrics:
        metrics[key] = []
    for key, value in step_metrics.items():
      metrics[key].append(value)
  return params, metrics


class AggregatorsTest(parameterized.TestCase):

  def test_aggregation_and_accumulation_match_standard(self):
    base_opt = alias.sgd(learning_rate=0.1)
    std_params, std_metrics = _train(base_opt)

    opt = aggregating.process(
        base.identity(),
        aggregating.average_incrementally_updates(
            per_elt_axis=0, accumulation_steps=1
        ),
        base_opt,
    )
    agg_params, agg_metrics = _train(opt, 1)
    device_type = jax.devices()[0].platform
    rtol = 5 * 1e-3 if device_type == 'tpu' else 1e-5
    with self.subTest('aggregation matches standard'):
      test_utils.assert_trees_all_close(std_params, agg_params, rtol=rtol)
      test_utils.assert_trees_all_close(
          std_metrics['full_batch_loss'],
          agg_metrics['full_batch_loss'],
          rtol=rtol,
      )

    opt = aggregating.process(
        base.identity(),
        aggregating.average_incrementally_updates(
            per_elt_axis=None, accumulation_steps=2
        ),
        base_opt,
    )
    acc_params, acc_metrics = _train(opt, accumulation_steps=2)

    with self.subTest('accumulation matches standard'):
      test_utils.assert_trees_all_close(std_params, acc_params)
      test_utils.assert_trees_all_close(
          std_metrics['full_batch_loss'], acc_metrics['full_batch_loss'][::2]
      )

    opt = aggregating.process(
        base.identity(),
        aggregating.average_incrementally_updates(
            per_elt_axis=0, accumulation_steps=2
        ),
        base_opt,
    )
    agg_acc_params, agg_acc_metrics = _train(opt, accumulation_steps=2)

    with self.subTest('aggregation and accumulation match standard'):
      test_utils.assert_trees_all_close(std_params, agg_acc_params)
      test_utils.assert_trees_all_close(
          std_metrics['full_batch_loss'],
          agg_acc_metrics['full_batch_loss'][::2],
      )

  @parameterized.product(ema_decay=[0.0, 0.9])
  def test_mean_variance_ema_match_standard(self, ema_decay: float = 0.99):
    base_opt = alias.sgd(learning_rate=0.1)
    std_params, std_metrics = _train(base_opt)

    opt = aggregating.add_mean_variance_to_opt(base_opt, ema_decay)
    mean_var_agg_params, mean_var_agg_metrics = _train(
        opt, metrics_ema_decay=ema_decay
    )

    device_type = jax.devices()[0].platform
    rtol = 5 * 1e-3 if device_type == 'tpu' else 1e-5
    with self.subTest(
        'mean variance ema with aggregation training matches standard'
    ):
      test_utils.assert_trees_all_close(
          std_params, mean_var_agg_params, rtol=rtol)
      test_utils.assert_trees_all_close(
          std_metrics['full_batch_loss'],
          mean_var_agg_metrics['full_batch_loss'],
          rtol=rtol,
      )
    with self.subTest('monitored mean grads ema matches true mean grads ema'):
      test_utils.assert_trees_all_close(
          mean_var_agg_metrics['true_mean_grads_ema'],
          mean_var_agg_metrics['mean_grads_ema'],
          rtol=rtol,
      )
    with self.subTest('monitored var grads ema matches true var grads ema'):
      test_utils.assert_trees_all_close(
          mean_var_agg_metrics['true_var_grads_ema'],
          mean_var_agg_metrics['var_grads_ema'],
          rtol=rtol,
      )

    opt = aggregating.add_mean_variance_to_opt(
        base_opt, ema_decay, accumulation_steps=2
    )
    mean_var_acc_params, mean_var_acc_metrics = _train(
        opt, accumulation_steps=2
    )
    with self.subTest(
        'mean variance ema with accumulation training matches standard'
    ):
      test_utils.assert_trees_all_close(
          std_params, mean_var_acc_params, rtol=rtol)
      test_utils.assert_trees_all_close(
          std_metrics['full_batch_loss'],
          mean_var_acc_metrics['full_batch_loss'][::2],
          rtol=rtol,
      )

    with self.subTest(
        'var grads ema with accumulation matches var grads ema with aggregation'
    ):
      test_utils.assert_trees_all_close(
          mean_var_agg_metrics['var_grads_ema'],
          mean_var_acc_metrics['var_grads_ema'][1::2],
      )


if __name__ == '__main__':
  absltest.main()
