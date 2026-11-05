# MLP MNIST

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/mlp_mnist.ipynb)

This notebook trains a simple Multilayer Perceptron (MLP) classifier for hand-written digit recognition (MNIST dataset).

To run the colab locally you need install the `grain`, `tensorflow-datasets` packages via `pip`.

```python
from typing import Sequence
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx

import grain
import tensorflow_datasets as tfds
```

```python
# @markdown The learning rate for the optimizer:
LEARNING_RATE = 0.002 # @param{type:"number"}
# @markdown Number of samples in each batch:
BATCH_SIZE = 128 # @param{type:"integer"}
# @markdown Total number of epochs to train for:
N_EPOCHS = 1 # @param{type:"integer"}
# Number of classes (digits 0-9)
N_TARGETS = 10
# Input size (MNIST images are 28x28 pixels)
IMG_SIZE = 28 * 28
# Directory for storing the dataset
DATA_DIR = '/tmp/mnist_dataset'
```

## Data Loading

MNIST is a dataset of 28x28 images with 1 channel. We now load the dataset using `tensorflow_datasets`, convert to `grain` dataset using `grain.MapDataset` and apply min-max normalization to images, shuffle the data in the train set and create batches of size `BATCH_SIZE`.

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

## Define MLP Model

The data is ready! Next let's define a model. Optax is agnostic to which (if any) neural network library is used. Here we use Flax NNX to implement a simple MLP.

```python
class MLP(nnx.Module):
  """A simple multilayer perceptron model for image classification."""
  def __init__(self, num_inputs: int, num_classes: int, hidden_sizes:
               Sequence[int], *, rngs: nnx.Rngs):
    self.hidden_sizes = hidden_sizes
    self.layer1 = nnx.Linear(num_inputs, self.hidden_sizes[0], rngs=rngs)
    self.layer2 = nnx.Linear(self.hidden_sizes[0], self.hidden_sizes[1],
                             rngs=rngs)
    self.layer_out = nnx.Linear(self.hidden_sizes[1], num_classes, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.layer1(x))
    x = nnx.relu(self.layer2(x))
    x = self.layer_out(x)
    return x

def compute_loss_and_accuracy(model, batch):
    inputs, labels = batch
    logits = model(inputs)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy

@nnx.jit
def train_step(model, optimizer, batch):
    """Performs a one step update."""
    grad_fn = nnx.value_and_grad(compute_loss_and_accuracy, has_aux=True)
    (loss, accuracy), grads = grad_fn(model, batch)

    # In-place update of the model parameters
    optimizer.update(model, grads)

    return loss, {"accuracy": accuracy}

@nnx.jit
def eval_step(model, batch):
    loss, accuracy = compute_loss_and_accuracy(model, batch)
    return loss, {"accuracy": accuracy}

def flatten_batches(dataloader):
    """Helper to flatten images to (BATCH_SIZE, 784)."""
    for inputs, labels in dataloader:
        yield inputs.reshape((inputs.shape[0], -1)), labels
```

Next we need to initialize network parameters and solver state. We also define a convenience function `dataset_stats` that we'll call once per epoch to collect the loss and accuracy of our solver over the test set.

```python
rngs = nnx.Rngs(0)

# Create the Model by flattening the tuple IMG_SIZE to an integer
flattened_img_size = int(np.prod(IMG_SIZE))
model = MLP(num_inputs=flattened_img_size, num_classes=N_TARGETS,
            hidden_sizes=[1000, 1000], rngs=rngs)

# Create the Optimizer
solver = optax.adam(LEARNING_RATE)
optimizer = nnx.Optimizer(model, solver, wrt=nnx.Param)

def dataset_stats(model, data_loader):
    """Computes loss and accuracy over the dataset."""
    all_accuracy = []
    all_loss = []
    for batch in data_loader:
        loss, aux = eval_step(model, batch)
        all_loss.append(loss)
        all_accuracy.append(aux["accuracy"])
    return {"loss": np.mean(all_loss), "accuracy": np.mean(all_accuracy)}
```

## Training Loop
Finally, we do the actual training. The next cell train the model for `N_EPOCHS`. Within each epoch we iterate over the batched loader `train_loader_batched`, and once per epoch we also compute the test set accuracy and loss.

```python
train_accuracy = []
train_losses = []
test_accuracy = []
test_losses = []

# Computes test set accuracy at initialization.
test_stats = dataset_stats(model, flatten_batches(test_loader_batched))
test_accuracy.append(test_stats["accuracy"])
test_losses.append(test_stats["loss"])

for epoch in range(N_EPOCHS):
    train_accuracy_epoch = []
    train_losses_epoch = []

    # Iterate over the training dataset
    for step, train_batch in enumerate(flatten_batches(train_loader_batched)):
        train_loss, train_aux = train_step(model, optimizer, train_batch)

        train_accuracy_epoch.append(train_aux["accuracy"])
        train_losses_epoch.append(train_loss)

        if step % 20 == 0:
            print(
                f"Step {step}, train loss: {train_loss:.4f}, train accuracy:"
                f" {train_aux['accuracy']:.2f}")

    # Record training stats
    train_accuracy.append(np.mean(train_accuracy_epoch))
    train_losses.append(np.mean(train_losses_epoch))

    # Evaluate on test set
    test_stats = dataset_stats(model, flatten_batches(test_loader_batched))
    test_accuracy.append(test_stats["accuracy"])
    test_losses.append(test_stats["loss"])
```

```python
f"Improved test accuracy from {test_accuracy[0]:.2f} to {test_accuracy[-1]:.2f}"
```
