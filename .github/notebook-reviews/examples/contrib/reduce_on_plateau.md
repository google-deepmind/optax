# Reduce on Plateau Learning Rate Scheduler

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/contrib/reduce_on_plateau.ipynb)

In this notebook, we explore the power of {py:func}`optax.contrib.reduce_on_plateau` scheduler, which reduces the learning rate when a metric has stopped improving. We will be solving a classification task by training a simple Multilayer Perceptron (MLP) on the fashion MNIST dataset.

To run the colab locally you need install the
`grain`, `tensorflow-datasets` packages via `pip`.

```python
from typing import Sequence
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import grain
import tensorflow_datasets as tfds

import optax
import optax.tree
from optax import contrib

# Show on which platform JAX is running.
print("JAX running on", jax.devices()[0].platform.upper())
```

## Data and model setup


Fashion MNIST is a dataset of 28x28 grayscale image, associated with a label from 10 classes. We now load the dataset using `tensorflow_datasets`, apply min-max normalization to the images, shuffle the data in the train set and create batches of size `BATCH_SIZE`.


```python
# @markdown Number of samples in each batch:
BATCH_SIZE = 128  # @param{type:"integer"}
```

```python
train_source, test_source = tfds.data_source("fashion_mnist", split=["train", "test"])
train_loader = grain.MapDataset.source(train_source)
test_loader = grain.MapDataset.source(test_source)

train_loader_batched = (
    train_loader
    .shuffle(seed=45)
    .map(lambda x: (x["image"] / 255., x["label"]))
    .batch(BATCH_SIZE, drop_remainder=True)
)
test_loader_batched = (
    test_loader
    .map(lambda x: (x["image"] / 255., x["label"]))
    .batch(BATCH_SIZE, drop_remainder=True)
)

IMG_SIZE = train_source.dataset_info.features["image"].shape
NUM_CLASSES = train_source.dataset_info.features["label"].num_classes
```

The data is ready! Next let's define a model. Optax is agnostic to which (if any) neural network library is used. Here we use Flax to implement a simple MLP.

```python
class MLP(nnx.Module):
  """A simple multilayer perceptron model for image classification."""
  def __init__(self, hidden_sizes: Sequence[int], num_classes: int, rngs: nnx.Rngs):
    # Initialize layers eagerly with nnx
    self.linear1 = nnx.Linear(28 * 28, hidden_sizes[0], rngs=rngs)
    self.linear2 = nnx.Linear(hidden_sizes[0], hidden_sizes[1], rngs=rngs)
    self.linear3 = nnx.Linear(hidden_sizes[1], num_classes, rngs=rngs)

  def __call__(self, x):
    # Flattens images in the batch.
    x = x.reshape((x.shape[0], -1))
    x = self.linear1(x)
    x = nnx.relu(x)
    x = self.linear2(x)
    x = nnx.relu(x)
    x = self.linear3(x)
    return x
```

```python
# Initialize the stateful NNX model
net = MLP(hidden_sizes=(1000, 1000), num_classes=NUM_CLASSES, rngs=nnx.Rngs(0))

# Split the model into a static graph definition and a dynamic parameter PyTree
graphdef, init_params = nnx.split(net, nnx.Param)

@jax.jit
def loss_accuracy(params, data):
  """Computes loss and accuracy over a mini-batch.

  Args:
    params: parameters of the model (nnx.Param state).
    data: tuple of (inputs, labels).

  Returns:
    loss: float
  """
  inputs, labels = data

  # Merge the graph definition and params back into a functional model for this step
  model = nnx.merge(graphdef, params)
  logits = model(inputs)

  loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=labels
  ).mean()
  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
  return loss, {"accuracy": accuracy}
```

Next we initialize network parameters.

```python
def dataset_stats(model, data_loader):
  """Computes loss and accuracy over the dataset `data_loader`."""
  all_accuracy = []
  all_loss = []
  for batch in data_loader:
    batch_loss, batch_aux = loss_accuracy(model, batch)
    all_loss.append(batch_loss)
    all_accuracy.append(batch_aux["accuracy"])
  return {"loss": np.mean(all_loss), "accuracy": np.mean(all_accuracy)}
```

## Reduce on average training loss

In this section, we consider an implementation that reduces the learning rate according to an average training loss value agglomerated for some accumulation size hyperparameter. In the next section, we consider an implementation that
reduces the learning rate according to the test loss.


In both examples, we consider an adam optimizer with a given learning rate that will be scaled by reduce on plateau over a total of 50 epochs.

```python
# @markdown Total number of epochs to train for:
N_EPOCHS = 50  # @param{type:"integer"}
# @markdown The base learning rate for the optimizer:
LEARNING_RATE = 0.01  # @param{type:"number"}
```

We set up the hyperparameters of reduce on plateau in this example.

```python
# @markdown Number of epochs with no improvement after which learning rate will be reduced:
PATIENCE = 5  # @param{type:"integer"}
# @markdown Number of epochs to wait before resuming normal operation after the learning rate reduction:
COOLDOWN = 0  # @param{type:"integer"}
# @markdown Factor by which to reduce the learning rate:
FACTOR = 0.5  # @param{type:"number"}
# @markdown Relative tolerance for measuring the new optimum:
RTOL = 1e-4  # @param{type:"number"}
# @markdown Number of iterations to accumulate an average value:
ACCUMULATION_SIZE = 200
```

We chain the base optimizer (adam) and the reduce on plateau transformation.

```python
opt = optax.chain(
    optax.adam(LEARNING_RATE),
    contrib.reduce_on_plateau(
        patience=PATIENCE,
        cooldown=COOLDOWN,
        factor=FACTOR,
        rtol=RTOL,
        accumulation_size=ACCUMULATION_SIZE,
    ),
)
opt_state = opt.init(nnx.state(net, nnx.Param))
```

In the training step, we feed the current value of the loss to the chained optimizer. This value is used to compute an average on ACCUMULATION_SIZE number of iterations by reduce on plateau.

```python
@jax.jit
def train_step(params, opt_state, batch):
  """Performs a one step update."""
  (value, aux), grad = jax.value_and_grad(loss_accuracy, has_aux=True)(
      params, batch
  )
  updates, opt_state = opt.update(grad, opt_state, params, value=value)
  params = optax.apply_updates(params, updates)
  return params, opt_state, value, aux


params = init_params

# Computes metrics at initialization.
train_stats = dataset_stats(params, test_loader_batched)
train_accuracy = [train_stats["accuracy"]]
train_losses = [train_stats['loss']]

test_stats = dataset_stats(params, test_loader_batched)
test_accuracy = [test_stats["accuracy"]]
test_losses = [test_stats["loss"]]

lr_scale_history = []
for epoch in range(N_EPOCHS):
  train_accuracy_epoch = []
  train_losses_epoch = []

  for _, train_batch in enumerate(train_loader_batched):
    params, opt_state, train_loss, train_aux = train_step(
        params, opt_state, train_batch
    )
    train_accuracy_epoch.append(train_aux["accuracy"])
    train_losses_epoch.append(train_loss)

  mean_train_accuracy = np.mean(train_accuracy_epoch)
  mean_train_loss = np.mean(train_losses_epoch)

  # fetch the scaling factor from the reduce_on_plateau transform
  lr_scale = optax.tree.get(opt_state, "scale")
  lr_scale_history.append(lr_scale)

  train_accuracy.append(mean_train_accuracy)
  train_losses.append(mean_train_loss)

  test_stats = dataset_stats(params, test_loader_batched)
  test_accuracy.append(test_stats["accuracy"])
  test_losses.append(test_stats["loss"])
  print(
      f"Epoch {epoch + 1}/{N_EPOCHS}, mean train accuracy:"
      f" {mean_train_accuracy}, lr scale: {optax.tree.get(opt_state, 'scale')}"
  )
```

```python
def plot(
    lr_scale_history, train_losses, train_accuracy, test_losses, test_accuracy
  ):
  plt.rcParams["figure.figsize"] = (20, 4.5)
  plt.rcParams.update({"font.size": 18})

  fig, axs = plt.subplots(ncols=5)

  axs[0].plot(lr_scale_history[1:], lw=3)
  axs[0].set_yscale('log')
  axs[0].set_title("LR Scale")
  axs[0].set_ylabel("LR Scale")
  axs[0].set_xlabel("Epoch")

  axs[1].plot(train_losses[1:], lw=3)
  axs[1].scatter(
      jnp.argmin(jnp.array(train_losses)),
      min(train_losses),
      label="Min",
      s=100,
  )
  axs[1].set_title("Train loss")
  axs[1].set_xlabel("Epoch")
  axs[1].set_ylabel("Train Loss")
  axs[1].legend(frameon=False)

  axs[2].plot(train_accuracy[1:], lw=3)
  axs[2].scatter(
      jnp.argmax(jnp.array(train_accuracy)),
      max(train_accuracy),
      label="Max",
      s=100,
  )
  axs[2].set_title("Train acc")
  axs[2].set_xlabel("Epoch")
  axs[2].set_ylabel("Train acc")
  axs[2].legend(frameon=False)

  axs[3].plot(test_losses[1:], lw=3)
  axs[3].scatter(
      jnp.argmin(jnp.array(test_losses)),
      min(test_losses),
      label="Min",
      s=100,
  )
  axs[3].set_title("Test loss")
  axs[3].set_xlabel("Epoch")
  axs[3].set_ylabel("Test Loss")
  axs[3].legend(frameon=False)

  axs[4].plot(test_accuracy[1:], lw=3)
  axs[4].scatter(
      jnp.argmax(jnp.array(test_accuracy)),
      max(test_accuracy),
      label="Max",
      s=100,
  )
  axs[4].set_title("Test acc")
  axs[4].set_ylabel("Test Acc")
  axs[4].legend(frameon=False)
  axs[4].set_xlabel("Epoch")

  plt.tight_layout()
  fig.show()
```

```python
plot(lr_scale_history, train_losses, train_accuracy, test_losses, test_accuracy)
```

## Reduce on test loss plateau

Here we consider an implementation that reduces the learning rate according to the test loss value. In this example, the accumulation size is just one as we manually gather the test loss outside of the transformation.

```python
# @markdown Number of epochs with no improvement after which learning rate will be reduced:
PATIENCE = 5  # @param{type:"integer"}
# @markdown Number of epochs to wait before resuming normal operation after the learning rate reduction:
COOLDOWN = 0  # @param{type:"integer"}
# @markdown Factor by which to reduce the learning rate:
FACTOR = 0.5  # @param{type:"number"}
# @markdown Relative tolerance for measuring the new optimum:
RTOL = 1e-4  # @param{type:"number"}
# @markdown Number of iterations to accumulate an average value:
ACCUMULATION_SIZE = 1
```

The base optimizer is still adam.

```python
solver = optax.adam(LEARNING_RATE)
solver_state = solver.init(nnx.state(net, nnx.Param))
```

In the next cell, we initialize the `contrib.reduce_on_plateau` scheduler, which reduces learning rate when a monitored metric (the test loss here) has stopped improving. We will be using this scheduler to scale the updates, produced by the regular Adam optimizer.

Note that the initial scale for the scheduler is not explicitly set, so it will default to 1.0, which means there will be no scaling of the learning rate initially.

```python
transform = contrib.reduce_on_plateau(
    patience=PATIENCE,
    cooldown=COOLDOWN,
    factor=FACTOR,
    rtol=RTOL,
    accumulation_size=ACCUMULATION_SIZE
    )

# Creates initial state for `contrib.reduce_on_plateau` transformation.
transform_state = transform.init(nnx.state(net, nnx.Param))
transform_state
```

The next cell trains the model for `N_EPOCHS` epochs. At the end of each epoch, the learning rate scaling value is updated based on the loss computed on the test set.

```python
@jax.jit
def train_step(params, solver_state, transform_state, batch):
  """Performs a one step update."""
  (loss, aux), grad = jax.value_and_grad(loss_accuracy, has_aux=True)(
      params, batch
  )
  # Computes updates scaled by the learning rate that was used to initialize
  # the `solver`.
  updates, solver_state = solver.update(grad, solver_state, params)
  # Scales updates, produced by `solver`, by the scaling value.
  updates = optax.tree.scale(transform_state.scale, updates)
  params = optax.apply_updates(params, updates)
  return params, solver_state, loss, aux

params = nnx.state(net, nnx.Param)

# Computes metrics at initialization.
train_stats = dataset_stats(params, test_loader_batched)
train_accuracy = [train_stats["accuracy"]]
train_losses = [train_stats['loss']]

test_stats = dataset_stats(params, test_loader_batched)
test_accuracy = [test_stats["accuracy"]]
test_losses = [test_stats["loss"]]

lr_scale_history = [transform_state.scale]
for epoch in range(N_EPOCHS):
  train_accuracy_epoch = []
  train_losses_epoch = []

  for train_batch in train_loader_batched:
    params, solver_state, train_loss, train_aux = train_step(
        params, solver_state, transform_state, train_batch
    )
    train_accuracy_epoch.append(train_aux["accuracy"])
    train_losses_epoch.append(train_loss)

  mean_train_accuracy = np.mean(train_accuracy_epoch)
  mean_train_loss = np.mean(train_losses_epoch)

  # Adjusts the learning rate scaling value using the loss computed on the
  # test set.
  _, transform_state = transform.update(
      updates=params, state=transform_state, value=test_stats["loss"]
  )
  lr_scale_history.append(transform_state.scale)

  train_accuracy.append(mean_train_accuracy)
  train_losses.append(mean_train_loss)

  test_stats = dataset_stats(params, test_loader_batched)
  test_accuracy.append(test_stats["accuracy"])
  test_losses.append(test_stats["loss"])

  print(
      f"Epoch {epoch + 1}/{N_EPOCHS}, mean train accuracy:"
      f" {mean_train_accuracy}, lr scale: {transform_state.scale}"
  )
```

```python
plot(lr_scale_history, train_losses, train_accuracy, test_losses, test_accuracy)
```

```python

```
