---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="j_LlXHYcmRaC" -->
# Gradient Accumulation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/gradient_accumulation.ipynb)
<!-- #endregion -->

<!-- #region id="vQro77whXULU" -->
_Gradient accumulation_ is a technique where the gradients for several consecutive optimization steps are combined together, so that they can be applied at regular repeating intervals.

One example where this is useful is to simulate training with a larger batch size than would fit into the available device memory. Another example is in the context of multi-task learning, where batches for different tasks may be visited in a round-robin fashion. Gradient accumulation makes it possible to simulate training on one large batch containing all of the tasks together.

In this example, we give an example of implementing gradient accumulation using {py:func}`optax.MultiSteps`. We start by bringing in some imports and defining some type annotations.
<!-- #endregion -->

```python id="9cu0kFNrnJj7"
from typing import Iterable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

```

<!-- #region id="TIKfATeXnW3B" -->
The following implements a network and loss function that could be used in an image classification problem.
<!-- #endregion -->

```python id="bJ1RWa4rnZmR"
class MLP(nn.Module):
  """A simple multilayer perceptron model."""

  @nn.compact
  def __call__(self, x):
    # Flattens inputs in the batch.
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(features=512)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

net = MLP()

def loss_fn(params, batch):
  """Computes loss over a mini-batch.
  """
  logits = net.apply(params, batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label']
  ).mean()
  return loss
```

<!-- #region id="FDq-pRJGnksx" -->
We implement a training loop to perform gradient descent as follows.
<!-- #endregion -->

```python id="uqKt4aBJXiBj"
def build_train_step(optimizer: optax.GradientTransformation):
  """Builds a function for executing a single step in the optimization."""

  @jax.jit
  def update(params, opt_state, batch):
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

  return update


def fit(
    optimizer: optax.GradientTransformation,
    params: optax.Params,
    batches: Iterable[dict[str, jnp.ndarray]],
) -> optax.Params:
  """Executes a train loop over the train batches using the given optimizer."""

  train_step = build_train_step(optimizer)
  opt_state = optimizer.init(params)

  for batch in batches:
    params, opt_state = train_step(params, opt_state, batch)

  return params
```

<!-- #region id="pTaorGnceOGs" -->
The following generates some random image-like data to test with our networks. The shapes used here correspond to the shapes that might appear in an MNIST classifier.

We also initialize some parameters and a base optimizer to share through the following examples.
<!-- #endregion -->

```python id="yK75QHOML7h9"
EXAMPLES = jax.random.uniform(jax.random.PRNGKey(0), (9, 28, 28, 1))
LABELS = jax.random.randint(jax.random.PRNGKey(0), (9,), minval=0, maxval=10)

optimizer = optax.sgd(1e-4)
params = net.init(jax.random.PRNGKey(0), EXAMPLES)
```

<!-- #region id="luk1TPW6efgQ" -->
## Splitting updates for one batch over multiple steps


The following two snippets will compute numerically identical results, but with the difference that the second snippet will use gradient accumulation over three batches to mimic the first snippet, which performs a single step with one large batch.

We start with the snippet that runs a training loop over a single batch containing all examples,
<!-- #endregion -->

```python id="hyykkSEio0Tx"
new_params_single_batch = fit(
    optimizer,
    params,
    batches=[dict(image=EXAMPLES, label=LABELS),]
)
```

<!-- #region id="7qIpPp0Jo6WT" -->
In this second snippet, our training loop will execute three training steps that together also contain all of the examples. In this case, the optimizer is wrapped with `optax.MultiSteps`, with `every_k_schedule=3`. This means that instead of applying gradient updates directly, the raw gradients will be combined together until the third step, where the wrapped optimizer will be applied to the average over the raw gradients seen up until now. For the "interim" steps, the updates returned by the optimizer will be all-zeros, resulting in no change to the parameters during these steps.
<!-- #endregion -->

```python id="pV1yZRxio2LS"
new_params_gradient_accumulation = fit(
    optax.MultiSteps(optimizer, every_k_schedule=3),
    params,
    batches=[
        dict(image=EXAMPLES[0:3], label=LABELS[0:3]),
        dict(image=EXAMPLES[3:6], label=LABELS[3:6]),
        dict(image=EXAMPLES[6:9], label=LABELS[6:9]),
    ],
)
```

<!-- #region id="gu8JnqgKo9Jq" -->
We can now verify that both training loops compute identical results as follows.
<!-- #endregion -->

```python id="X2hWzwFkK43k"
def assert_trees_all_close(a, b):
  """Asserts that two pytrees of arrays are close within a tolerance."""
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_allclose(x, y, atol=1e-7)

assert_trees_all_close(
    new_params_single_batch,
    new_params_gradient_accumulation,
)
```

<!-- #region id="Ub0GHPvvhIKI" -->
## Interaction of {py:func}`optax.MultiStep` with schedules.

The snippet below is identical to the snippet above, except we additionally introduce a learning rate schedule. As above, the second call to `fit` is using gradient accumulation. Similarly to before, we find that both train loops compute compute identical outputs (up to numerical errors).

This happens because the learning rate schedule in {py:func}`optax.MultiStep` is only updated once for each of the _outer_ steps. In particular, the state of the inner optimizer is only updated each time `every_k_schedule` optimizer steps have been taken.
<!-- #endregion -->

```python id="o9CS96VjMuON"
learning_rate_schedule = optax.piecewise_constant_schedule(
    init_value=1.0,
    boundaries_and_scales={
        0: 1e-4,
        1: 1e-1,
    },
)

optimizer = optax.sgd(learning_rate_schedule)

new_params_single_batch = fit(
    optimizer,
    params,
    batches=[
        dict(image=EXAMPLES, label=LABELS),
    ],
)

new_params_gradient_accumulation = fit(
    optax.MultiSteps(optimizer, every_k_schedule=3),
    params,
    batches=[
        dict(image=EXAMPLES[0:3], label=LABELS[0:3]),
        dict(image=EXAMPLES[3:6], label=LABELS[3:6]),
        dict(image=EXAMPLES[6:9], label=LABELS[6:9]),
    ],
)

assert_trees_all_close(
    new_params_single_batch,
    new_params_gradient_accumulation,
)
```
