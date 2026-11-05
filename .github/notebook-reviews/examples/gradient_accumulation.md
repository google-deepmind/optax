# Gradient Accumulation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/gradient_accumulation.ipynb)


_Gradient accumulation_ is a technique where the gradients for several consecutive optimization steps are combined together, so that they can be applied at regular repeating intervals.

One example where this is useful is to simulate training with a larger batch size than would fit into the available device memory. Another example is in the context of multi-task learning, where batches for different tasks may be visited in a round-robin fashion. Gradient accumulation makes it possible to simulate training on one large batch containing all of the tasks together.

In this example, we give an example of implementing gradient accumulation using {py:func}`optax.MultiSteps`. We start by bringing in some imports and defining some type annotations.

```python
from typing import Iterable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

```

The following implements a network and loss function that could be used in an image classification problem.

```python
class MLP(nnx.Module):
  """A simple multilayer perceptron model using flax.nnx."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(28 * 28, 512, rngs=rngs)
    self.linear2 = nnx.Linear(512, 512, rngs=rngs)
    self.linear3 = nnx.Linear(512, 10, rngs=rngs)

  def __call__(self, x):
    # Flattens inputs in the batch.
    x = x.reshape((x.shape[0], -1))
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    x = jax.nn.relu(x)
    x = self.linear3(x)
    return x

def loss_fn(model, batch):
  """Computes loss over a mini-batch."""
  logits = model(batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label']
  ).mean()
  return loss
```

We implement a training loop to perform gradient descent as follows.

```python
def build_train_step():
  """Builds a function for executing a single step in the optimization."""

  @jax.jit
  def update(model, optimizer, batch):
    grads = nnx.grad(loss_fn)(model, batch)
    optimizer.update(model, grads)

  return update


def fit(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batches: Iterable[dict[str, jnp.ndarray]],
):
  """Executes a train loop over the train batches using the given optimizer."""

  train_step = build_train_step()

  for batch in batches:
    train_step(model, optimizer, batch)

```

The following generates some random image-like data to test with our networks. The shapes used here correspond to the shapes that might appear in an MNIST classifier.

We also initialize some parameters and a base optimizer to share through the following examples.

```python
EXAMPLES = jax.random.uniform(jax.random.PRNGKey(0), (9, 28, 28, 1))
LABELS = jax.random.randint(jax.random.PRNGKey(0), (9,), minval=0, maxval=10)

optimizer = optax.sgd(1e-4)
```

<!-- #region -->
## Splitting updates for one batch over multiple steps


The following two snippets will compute numerically identical results, but with the difference that the second snippet will use gradient accumulation over three batches to mimic the first snippet, which performs a single step with one large batch.

We start with the snippet that runs a training loop over a single batch containing all examples,
<!-- #endregion -->

```python
model_single_batch = MLP(nnx.Rngs(0))
opt_single_batch = nnx.Optimizer(model_single_batch, optimizer, wrt=nnx.Param)

new_params_single_batch = fit(
    model_single_batch,
    opt_single_batch,
    batches=[dict(image=EXAMPLES, label=LABELS),]
)
```

In this second snippet, our training loop will execute three training steps that together also contain all of the examples. In this case, the optimizer is wrapped with `optax.MultiSteps`, with `every_k_schedule=3`. This means that instead of applying gradient updates directly, the raw gradients will be combined together until the third step, where the wrapped optimizer will be applied to the average over the raw gradients seen up until now. For the "interim" steps, the updates returned by the optimizer will be all-zeros, resulting in no change to the parameters during these steps.

```python
model_gradient_accumulation = MLP(nnx.Rngs(0))
opt_gradient_accumulation = nnx.Optimizer(
    model_gradient_accumulation,
    optax.MultiSteps(optimizer, every_k_schedule=3),
    wrt=nnx.Param
)

new_params_gradient_accumulation = fit(
    model_gradient_accumulation,
    opt_gradient_accumulation,
    batches=[
        dict(image=EXAMPLES[0:3], label=LABELS[0:3]),
        dict(image=EXAMPLES[3:6], label=LABELS[3:6]),
        dict(image=EXAMPLES[6:9], label=LABELS[6:9]),
    ],
)
```

We can now verify that both training loops compute identical results as follows.

```python
def assert_trees_all_close(a, b):
  """Asserts that two pytrees of arrays are close within a tolerance."""
  for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
    np.testing.assert_allclose(x, y, atol=1e-7)

assert_trees_all_close(
    new_params_single_batch,
    new_params_gradient_accumulation,
)
```

## Interaction of {py:func}`optax.MultiStep` with schedules.

The snippet below is identical to the snippet above, except we additionally introduce a learning rate schedule. As above, the second call to `fit` is using gradient accumulation. Similarly to before, we find that both train loops compute compute identical outputs (up to numerical errors).

This happens because the learning rate schedule in {py:func}`optax.MultiStep` is only updated once for each of the _outer_ steps. In particular, the state of the inner optimizer is only updated each time `every_k_schedule` optimizer steps have been taken.

```python
learning_rate_schedule = optax.piecewise_constant_schedule(
    init_value=1.0,
    boundaries_and_scales={
        0: 1e-4,
        1: 1e-1,
    },
)

optimizer = optax.sgd(learning_rate_schedule)

model_single_batch = MLP(nnx.Rngs(0))
opt_single_batch = nnx.Optimizer(model_single_batch, optimizer, wrt=nnx.Param)

new_params_single_batch = fit(
    model_single_batch,
    opt_single_batch,
    batches=[
        dict(image=EXAMPLES, label=LABELS),
    ],
)

model_gradient_accumulation = MLP(nnx.Rngs(0))
opt_gradient_accumulation = nnx.Optimizer(
    model_gradient_accumulation,
    optax.MultiSteps(optimizer, every_k_schedule=3),
    wrt=nnx.Param
)

new_params_gradient_accumulation = fit(
    model_gradient_accumulation,
    opt_gradient_accumulation,
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
