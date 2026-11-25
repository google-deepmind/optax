# Lookahead Optimizer on MNIST

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/lookahead_mnist.ipynb)

This notebook trains a simple Convolution Neural Network (CNN) for hand-written digit recognition (MNIST dataset) using {py:func}`optax.lookahead`.

To run the colab locally you need install the
`grain`, `tensorflow-datasets` packages via `pip`.

```python
from functools import partial
from flax import nnx
import jax
import jax.numpy as jnp
import optax

import tensorflow_datasets as tfds
import grain
```

```python
# @markdown The learning rate for the fast optimizer:
FAST_LEARNING_RATE = 0.002 # @param{type:"number"}
# @markdown The learning rate for the slow optimizer:
SLOW_LEARNING_RATE = 0.1 # @param{type:"number"}
# @markdown Number of fast optimizer steps to take before synchronizing parameters:
SYNC_PERIOD = 5 # @param{type:"integer"}
# @markdown Number of samples in each batch:
BATCH_SIZE = 256 # @param{type:"integer"}
# @markdown Total number of epochs to train for:
N_EPOCHS = 1 # @param{type:"integer"}
```

MNIST is a dataset of 28x28 images with 1 channel. We now load the dataset using `tensorflow_datasets`, convert to grain dataset using `grain.MapDataset` and apply min-max normalization to images, shuffle the data in the train set and create batches of size `BATCH_SIZE`.


```python
train_source, test_source = tfds.data_source("mnist", split=["train", "test"])

IMG_SIZE = train_source.dataset_info.features["image"].shape
NUM_CLASSES = train_source.dataset_info.features["label"].num_classes

train_loader_batched = (
    grain.MapDataset.source(train_source)
    .shuffle(seed=45)
    .map(lambda x: (x["image"] / 255., x["label"]))
    .batch(BATCH_SIZE, drop_remainder=True)
)

test_loader_batched = (
    grain.MapDataset.source(test_source)
    .map(lambda x: (x["image"] / 255., x["label"]))
    .batch(BATCH_SIZE, drop_remainder=True)
)
```

The data is ready! Next let's define a model. Optax is agnostic to which (if any) neural network library is used. Here we use Flax to implement a simple CNN.

```python
class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, in_features, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(in_features=in_features,
                          out_features=32,
                          kernel_size=(3, 3),
                          rngs=rngs)
    self.pool1 = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.conv2 = nnx.Conv(in_features=32,
                          out_features=64,
                          kernel_size=(3, 3),
                          rngs=rngs)
    self.pool2 = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.dense1 = nnx.Linear(in_features=7*7*64, out_features=256, rngs=rngs)
    self.dense2 = nnx.Linear(in_features=256, out_features=10, rngs=rngs)

  def __call__(self, x):
    x = self.conv1(x)
    x = nnx.relu(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = nnx.relu(x)
    x = self.pool2(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = self.dense1(x)
    x = nnx.relu(x)
    x = self.dense2(x)
    return x
```

```python
net = CNN(IMG_SIZE[-1], nnx.Rngs(0))

@jax.jit
def predict(params, model, inputs):
  nnx.update(model, params)
  return model(inputs)


@jax.jit
def loss_accuracy(params, model, data):
  """Computes loss and accuracy over a mini-batch.

  Args:
    params: parameters of the model.
    model: Instance of the nnx.Module.
    data: tuple of (inputs, labels).

  Returns:
    loss: float
    accuracy: float
  """
  inputs, labels = data
  logits = predict(params, model, inputs)
  loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=labels
  ).mean()
  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
  return loss, {"accuracy": accuracy}
```

Next we need to initialize CNN parameters and solver state. We also define a convenience function `dataset_stats` that we'll call once per epoch to collect the loss and accuracy of our solver over the test set. We will be using the Lookahead optimizer.
Its wrapper keeps a pair of slow and fast parameters. To
initialize them, we create a pair of synchronized parameters from the
initial model parameters.


```python
fast_solver = optax.adam(FAST_LEARNING_RATE)
solver = optax.lookahead(fast_solver, SYNC_PERIOD, SLOW_LEARNING_RATE)
rng = jax.random.PRNGKey(0)
dummy_data = jnp.ones((1,) + IMG_SIZE, dtype=jnp.float32)

_ = net(dummy_data)
params = nnx.state(net, nnx.Param)

# Initializes the lookahead optimizer with the initial model parameters.
params = optax.LookaheadParams.init_synced(params)
solver_state = solver.init(params)

def dataset_stats(params, model, data_loader):
  """Computes loss and accuracy over the dataset `data_loader`."""
  all_accuracy = []
  all_loss = []
  for batch in data_loader:
    batch_loss, batch_aux = loss_accuracy(params, model, batch)
    all_loss.append(batch_loss)
    all_accuracy.append(batch_aux["accuracy"])
  return {"loss": jnp.mean(jnp.array(all_loss)),
          "accuracy": jnp.mean(jnp.array(all_accuracy))}
```

Finally, we do the actual training. The next cell train the model for  `N_EPOCHS`. Within each epoch we iterate over the batched loader `train_loader_batched`, and once per epoch we also compute the test set accuracy and loss.

```python
train_accuracy = []
train_losses = []

# Computes test set accuracy at initialization.
test_stats = dataset_stats(params.slow, net, test_loader_batched)
test_accuracy = [test_stats["accuracy"]]
test_losses = [test_stats["loss"]]


@jax.jit
def train_step(params, model, solver_state, batch):
  # Performs a one step update.
  (loss, aux), grad = jax.value_and_grad(loss_accuracy, has_aux=True)(
      params.fast, model, batch
  )
  updates, solver_state = solver.update(grad, solver_state, params)
  params = optax.apply_updates(params, updates)
  return params, solver_state, loss, aux


for epoch in range(N_EPOCHS):
  train_accuracy_epoch = []
  train_losses_epoch = []

  for step, train_batch in enumerate(train_loader_batched):
    params, solver_state, train_loss, train_aux = train_step(
        params, net, solver_state, train_batch
    )
    train_accuracy_epoch.append(train_aux["accuracy"])
    train_losses_epoch.append(train_loss)
    if step % 20 == 0:
      print(
          f"step {step}, train loss: {train_loss:.2e}, train accuracy:"
          f" {train_aux['accuracy']:.2f}"
      )

  # Validation is done on the slow lookahead parameters.
  test_stats = dataset_stats(params.slow, net, test_loader_batched)
  test_accuracy.append(test_stats["accuracy"])
  test_losses.append(test_stats["loss"])
  train_accuracy.append(jnp.mean(jnp.array(train_accuracy_epoch)))
  train_losses.append(jnp.mean(jnp.array(train_losses_epoch)))
```

```python
f"Improved accuracy on test DS from {test_accuracy[0]} to {test_accuracy[-1]}"
```
