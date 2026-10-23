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
# MLP MNIST

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/mlp_mnist.ipynb)

This notebook trains a simple Multilayer Perceptron (MLP) classifier for hand-written digit recognition (MNIST dataset).

To run the colab locally you need install `grain` via `pip`.
<!-- #endregion -->

```python id="9cu0kFNrnJj7"
from typing import Sequence
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx

import grain.python as pygrain
from torchvision.datasets import MNIST
import torchvision.transforms as T
```

```python id="2Adl_l_uZs1d"
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

<!-- #region id="ZZej3FcOhuRE" -->
## Data Loading

MNIST is a dataset of 28x28 images with 1 channel. We now load the dataset using `torchvision`, apply min-max normalization to the images, shuffle the data in the train set and create batches of size `BATCH_SIZE` using `grain`.
<!-- #endregion -->

```python id="xPZ0paOehHWg"
# Define the transformation
torch_transforms = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.ravel()),  # Flattens to (784,)
])

class Dataset:
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.load_data()

    def load_data(self):
        self.dataset = MNIST(self.data_dir, download=True, train=self.train,
                             transform=torch_transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return np.array(img, dtype=np.float32), label
```

<!-- #region id="idJlmlimHhsA" -->
Initialize the datasets
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="nLa9CBOtmHeq" outputId="354738d2-6a12-451e-89b4-b72805269936"
mnist_dataset_train = Dataset(DATA_DIR, train=True)
mnist_dataset_test = Dataset(DATA_DIR, train=False)

print(f"Train dataset size: {len(mnist_dataset_train)}")
print(f"Test dataset size: {len(mnist_dataset_test)}")
```

<!-- #region id="OkvjPx-AmR6n" -->
Initialize PyGrain DataLoaders
<!-- #endregion -->

```python id="iMtdFrd1FRAp"
train_sampler = pygrain.SequentialSampler(
    num_records=len(mnist_dataset_train),
    shard_options=pygrain.NoSharding()
)

train_loader_batched = pygrain.DataLoader(
    data_source=mnist_dataset_train,
    sampler=train_sampler,
    operations=[pygrain.Batch(batch_size=BATCH_SIZE, drop_remainder=True)],
)

test_sampler = pygrain.SequentialSampler(
    num_records=len(mnist_dataset_test),
    shard_options=pygrain.NoSharding()
)

test_loader = pygrain.DataLoader(
    data_source=mnist_dataset_test,
    sampler=test_sampler,
    operations=[pygrain.Batch(batch_size=BATCH_SIZE, drop_remainder=True)],
)
```

<!-- #region id="AXHuUNZhmmSB" -->
## Define MLP Model

The data is ready! Next let's define a model. Optax is agnostic to which (if any) neural network library is used. Here we use Flax NNX to implement a simple MLP.
<!-- #endregion -->

```python id="DVqFRcLqRklV"
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
    optimizer.update(grads)

    return loss, {"accuracy": accuracy}

@nnx.jit
def eval_step(model, batch):
    loss, accuracy = compute_loss_and_accuracy(model, batch)
    return loss, {"accuracy": accuracy}
```

<!-- #region id="peEolvEdnQ7n" -->
Next we need to initialize network parameters and solver state. We also define a convenience function `dataset_stats` that we'll call once per epoch to collect the loss and accuracy of our solver over the test set.
<!-- #endregion -->

```python id="BwWtqtepRnFE"
# Initialize RNGs
rngs = nnx.Rngs(0)

# Create the Model
model = MLP(num_inputs=IMG_SIZE, num_classes=N_TARGETS,
            hidden_sizes=[1000, 1000], rngs=rngs)

# Create the Optimizer
solver = optax.adam(LEARNING_RATE)
optimizer = nnx.Optimizer(model, solver)

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

<!-- #region id="hhqJY9GPlAZS" -->
## Training Loop
Finally, we do the actual training. The next cell train the model for `N_EPOCHS`. Within each epoch we iterate over the batched loader `train_loader_batched`, and once per epoch we also compute the test set accuracy and loss.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="19u_O6l9ydLM" outputId="a5b6b415-b117-4439-ea6b-4d9a5b2eeae7"
train_accuracy = []
train_losses = []
test_accuracy = []
test_losses = []

# Computes test set accuracy at initialization.
test_stats = dataset_stats(model, test_loader)
test_accuracy.append(test_stats["accuracy"])
test_losses.append(test_stats["loss"])

for epoch in range(N_EPOCHS):
    train_accuracy_epoch = []
    train_losses_epoch = []

    # Iterate over the training dataset
    for step, train_batch in enumerate(train_loader_batched):
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
    test_stats = dataset_stats(model, test_loader)
    test_accuracy.append(test_stats["accuracy"])
    test_losses.append(test_stats["loss"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 36} id="yyS1oRZBtytP" outputId="2a186ba1-a39b-4226-d673-9822bf63d508"
f"Improved test accuracy from {test_accuracy[0]:.2f} to {test_accuracy[-1]:.2f}"
```
