# ResNet on CIFAR10 with Flax NNX and Optax.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/cifar10_resnet.ipynb)

This notebook trains a residual network (ResNet) with **Flax NNX** and **Optax** on CIFAR10.

It demonstrates:
1. Loading data via `tensorflow_datasets`.
2. Data augmentation using `albumentations`.
3. Defining a ResNet model using the `flax.nnx`.
4. Computing loss and regularization using `optax`.
5. Training with `nnx.Optimizer`.

```{code-cell} ipython3
import functools
from typing import Any, Callable, Dict, Sequence, Tuple, Optional
from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import albumentations as A
from matplotlib import pyplot as plt

# Show on which platform JAX is running.
print("JAX running on", jax.devices()[0].platform.upper())
```

```{code-cell} ipython3
# @markdown Total number of epochs to train for:
MAX_EPOCHS = 50  # @param{type:"integer"}
# @markdown Number of samples in each batch:
BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown The initial learning rate for the optimizer:
PEAK_LR = 0.12  # @param{type:"number"}
# @markdown The model architecture for the neural network. Can be one of `'resnet1'`, `'resnet18'`, `'resnet34'`, `'resnet50'`, `'resnet101'`, `'resnet152'`, `'resnet200'`:
MODEL = "resnet18"  # @param{type:"string"}
# @markdown The dataset to use. Could be either `'cifar10'` or `'cifar100'`:
DATASET = "cifar10"  # @param{type:"string"}
# @markdown The amount of L2 regularization (aka weight decay) to use:
L2_REG = 1e-4  # @param{type:"number"}
```

## Data Loading
CIFAR10 and CIFAR100 are composed of 32x32 images with 3 channels (RGB). We'll now load the dataset using `tensorflow_datasets` and display a few of the first samples.

```{code-cell} ipython3
(train_ds_raw, test_ds_raw), info = tfds.load(
    DATASET, split=["train", "test"], as_supervised=True, with_info=True
)

# Convert to list of dictionaries for easy access
train_data_list = list(tfds.as_numpy(train_ds_raw))
test_data_list = list(tfds.as_numpy(test_ds_raw))

print(f"Loaded {len(train_data_list)} training images and {len(test_data_list)} test images.")

NUM_CLASSES = info.features["label"].num_classes
IMG_SIZE = info.features["image"].shape

def plot_sample_images(loader):
  loader_iter = iter(loader)
  _, axes = plt.subplots(nrows=4, ncols=5, figsize=(6, 4))
  for i in range(4):
    for j in range(5):
      image, label = next(loader_iter)
      axes[i, j].imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
      axes[i, j].set_axis_off()
      axes[i, j].set_title(
          info.features["label"].names[label], fontsize=10, y=0.9
      )

plot_sample_images(train_ds_raw)
```

## Data Augmentation
The accuracy of the model can be improved significantly through data augmentation. We use `albumentations` to apply random padding, cropping, horizontal flips, and normalization.

In the next cell we define these transformations and a custom generator to yield batches.

```{code-cell} ipython3
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

train_transforms = A.Compose([
    A.PadIfNeeded(min_height=40, min_width=40, p=1.0),
    A.RandomCrop(width=32, height=32, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

val_transforms = A.Compose([
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# Batch Generator
def get_batch_generator(dataset_list, transforms, batch_size, shuffle=True):
    indices = np.arange(len(dataset_list))

    def generator():
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(dataset_list), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Drop remainder if batch is too small (training only generally)
            if len(batch_indices) < batch_size and shuffle:
                continue

            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                sample = dataset_list[idx]
                # Apply augmentation
                aug_img = transforms(image=sample[0])['image']
                batch_images.append(aug_img)
                batch_labels.append(sample[1])

            yield {
                'image': np.array(batch_images),
                'label': np.array(batch_labels)
            }

    return generator
```

Displaying augmented sample images (restored from normalization).

```{code-cell} ipython3
TEMP_BATCH_SIZE = 20
augmented_generator_func = get_batch_generator(train_data_list, train_transforms, TEMP_BATCH_SIZE, shuffle=False)
augmented_batch = next(augmented_generator_func())

# image = (image * std) + mean
mean = np.array(CIFAR10_MEAN)
std = np.array(CIFAR10_STD)
images_restored = augmented_batch['image'] * std + mean
images_restored = np.clip(images_restored, 0, 1)

augmented_images_labels = [
    (images_restored[i], augmented_batch['label'][i])
    for i in range(len(images_restored))
]

plot_sample_images(augmented_images_labels)
```

We now instantiate generators that will be used during the training loop.

```{code-cell} ipython3
train_loader_fn = get_batch_generator(train_data_list, train_transforms, BATCH_SIZE, shuffle=True)
test_loader_fn = get_batch_generator(test_data_list, val_transforms, BATCH_SIZE, shuffle=False)
```

## Model Definition using Flax NNX
We implement the ResNet architecture inheriting from `nnx.Module`.

Since **Flax NNX** modules are Python objects that hold their own state (parameters and batch norm statistics), we handle the `train` boolean flag to switch between updating Batch Norm statistics (during training) or using the running average (during inference).

```{code-cell} ipython3
class ResNetBlock(nnx.Module):
  """ResNet block."""
  def __init__(
      self,
      in_features: int,
      filters: int,
      stride: int = 1,
      rngs: nnx.Rngs = None
  ):
    self.conv1 = nnx.Conv(in_features, filters, kernel_size=(3, 3), strides=stride, use_bias=False, rngs=rngs)
    self.bn1 = nnx.BatchNorm(filters, momentum=0.9, epsilon=1e-5, rngs=rngs)
    self.conv2 = nnx.Conv(filters, filters, kernel_size=(3, 3), use_bias=False, rngs=rngs)
    self.bn2 = nnx.BatchNorm(filters, momentum=0.9, epsilon=1e-5, rngs=rngs)

    self.proj = None
    if stride != 1 or in_features != filters:
        self.proj = nnx.Sequential(
            nnx.Conv(in_features, filters, kernel_size=(1, 1), strides=stride, use_bias=False, rngs=rngs),
            nnx.BatchNorm(filters, momentum=0.9, epsilon=1e-5, rngs=rngs)
        )

  def __call__(self, x, train: bool):
    residual = x
    y = self.conv1(x)
    y = self.bn1(y, use_running_average=not train)
    y = nnx.relu(y)
    y = self.conv2(y)
    y = self.bn2(y, use_running_average=not train)

    if self.proj is not None:
        residual = self.proj.layers[0](residual)
        residual = self.proj.layers[1](residual, use_running_average=not train)

    return nnx.relu(residual + y)


class BottleneckResNetBlock(nnx.Module):
  """Bottleneck ResNet block."""
  def __init__(
      self,
      in_features: int,
      filters: int,
      stride: int = 1,
      rngs: nnx.Rngs = None
  ):
    self.conv1 = nnx.Conv(in_features, filters, kernel_size=(1, 1), use_bias=False, rngs=rngs)
    self.bn1 = nnx.BatchNorm(filters, momentum=0.9, epsilon=1e-5, rngs=rngs)
    self.conv2 = nnx.Conv(filters, filters, kernel_size=(3, 3), strides=stride, use_bias=False, rngs=rngs)
    self.bn2 = nnx.BatchNorm(filters, momentum=0.9, epsilon=1e-5, rngs=rngs)
    self.conv3 = nnx.Conv(filters, filters * 4, kernel_size=(1, 1), use_bias=False, rngs=rngs)
    self.bn3 = nnx.BatchNorm(filters * 4, momentum=0.9, epsilon=1e-5, rngs=rngs)

    self.proj = None
    if stride != 1 or in_features != filters * 4:
        self.proj = nnx.Sequential(
            nnx.Conv(in_features, filters * 4, kernel_size=(1, 1), strides=stride, use_bias=False, rngs=rngs),
            nnx.BatchNorm(filters * 4, momentum=0.9, epsilon=1e-5, rngs=rngs)
        )

  def __call__(self, x, train: bool):
    residual = x
    y = self.conv1(x)
    y = self.bn1(y, use_running_average=not train)
    y = nnx.relu(y)
    y = self.conv2(y)
    y = self.bn2(y, use_running_average=not train)
    y = nnx.relu(y)
    y = self.conv3(y)
    y = self.bn3(y, use_running_average=not train)

    if self.proj is not None:
        residual = self.proj.layers[0](residual)
        residual = self.proj.layers[1](residual, use_running_average=not train)

    return nnx.relu(residual + y)


class ResNet(nnx.Module):
  """ResNetV1."""
  def __init__(
      self,
      stage_sizes: Sequence[int],
      block_cls: Any,
      num_classes: int,
      num_filters: int = 64,
      rngs: nnx.Rngs = None,
  ):
    self.conv1 = nnx.Conv(3, num_filters, kernel_size=(3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
    self.bn1 = nnx.BatchNorm(num_filters, momentum=0.9, epsilon=1e-5, rngs=rngs)

    self.blocks = []
    current_filters = num_filters

    for i, block_size in enumerate(stage_sizes):
      for j in range(block_size):
        stride = 2 if i > 0 and j == 0 else 1
        # Determine filter size expansion
        filters = num_filters * 2**i

        block = block_cls(current_filters, filters, stride=stride, rngs=rngs)
        self.blocks.append(block)

        if block_cls == BottleneckResNetBlock:
             current_filters = filters * 4
        else:
             current_filters = filters

    self.linear = nnx.Linear(current_filters, num_classes, rngs=rngs)

  def __call__(self, x, train: bool = True):
    x = self.conv1(x)
    x = self.bn1(x, use_running_average=not train)
    x = nnx.relu(x)

    for block in self.blocks:
      x = block(x, train=train)

    x = jnp.mean(x, axis=(1, 2))
    x = self.linear(x)
    return x

ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)
```

Note that we're overwriting some of the default parameters in this implementation, such as the `kernel_size` and `strides` of the convolutions. The default values of (7, 7) and 2 respectively are too large for the small 32x32 images in this dataset, so we reduce these parameters to (3, 3) and 1 respectively.

```{code-cell} ipython3
RESNET_CONSTRUCTOR = {
    "resnet1": ResNet1,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnet200": ResNet200,
}

# Initialize the model
rngs = nnx.Rngs(0)
model = RESNET_CONSTRUCTOR[MODEL](num_classes=NUM_CLASSES, rngs=rngs)

# Visual check of structure
nnx.display(model)
```

```{code-cell} ipython3
iter_per_epoch_train = info.splits["train"].num_examples // BATCH_SIZE
lr_schedule = optax.linear_onecycle_schedule(
    MAX_EPOCHS * iter_per_epoch_train, PEAK_LR
)

iterate_subsample = np.linspace(0, MAX_EPOCHS * iter_per_epoch_train, 100)
plt.plot(
    np.linspace(0, MAX_EPOCHS, len(iterate_subsample)),
    [lr_schedule(i) for i in iterate_subsample],
    lw=3,
)
plt.title("Learning rate")
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.grid()
plt.xlim((0, MAX_EPOCHS))
plt.show()
```

## Optimization and Training Loop
We use `nnx.Optimizer` to manage the optimization state.

In the `compute_loss` function, we leverage `optax` for both the cross-entropy loss (`optax.softmax_cross_entropy_with_integer_labels`) and L2 regularization (`optax.l2_loss`).

The `train_step` uses `nnx.jit` and `nnx.value_and_grad` to compute gradients and update the model in-place.

```{code-cell} ipython3
optimizer = nnx.Optimizer(model, optax.sgd(lr_schedule, momentum=0.9, nesterov=False), wrt=nnx.Param)

def compute_loss(model, batch, training: bool):
    logits = model(batch['image'], train=training)
    labels = batch['label']

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()

    params = nnx.state(model, nnx.Param)
    l2_penalty = sum(
        jnp.sum(optax.l2_loss(p))
        for p in jax.tree_util.tree_leaves(params)
    )

    total_loss = loss + L2_REG * l2_penalty
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return total_loss, accuracy

@nnx.jit
def train_step(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, accuracy), grads = grad_fn(model, batch, training=True)
    optimizer.update(model, grads)
    return loss, accuracy

@nnx.jit
def eval_step(model, batch):
    loss, accuracy = compute_loss(model, batch, training=False)
    return loss, accuracy

def evaluate(model, data_loader):
    losses, accuracies = [], []
    for batch in data_loader:
        loss, acc = eval_step(model, batch)
        losses.append(loss)
        accuracies.append(acc)
    return np.mean(losses), np.mean(accuracies)
```

## Run Training

Finally, we do the actual training. The next cell performs `'MAX_EPOCHS'` epochs of training. Within each epoch we iterate over the batched loader, and once per epoch we also compute the test set accuracy and loss.

```{code-cell} ipython3
train_accuracy = []
train_losses = []
test_accuracy = []
test_losses = []

# Computes test set accuracy at initialization.
test_loss, test_acc = evaluate(model, test_loader_fn())
test_accuracy.append(test_acc)
test_losses.append(test_loss)

# Executes a training loop.
for epoch in range(MAX_EPOCHS):
    train_accuracy_epoch = []
    train_losses_epoch = []

    # Train for one epoch
    for train_batch in train_loader_fn():
        train_loss, train_acc = train_step(model, optimizer, train_batch)
        train_accuracy_epoch.append(train_acc)
        train_losses_epoch.append(train_loss)

    # Calculate average train metrics for the epoch
    train_accuracy.append(np.mean(train_accuracy_epoch))
    train_losses.append(np.mean(train_losses_epoch))

    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader_fn())
    test_accuracy.append(test_acc)
    test_losses.append(test_loss)

    # Prints accuracy every 10 epochs.
    if epoch % 10 == 0:
      print(f"Epoch: {epoch}")
      print(f"Test set accuracy: {test_accuracy[-1]:.4f}")
      print(f"Train set accuracy: {train_accuracy[-1]:.4f}")
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plt.suptitle(f"{MODEL} on {DATASET}", fontsize=20)

ax1.plot(
    test_accuracy,
    lw=3,
    marker="s",
    markevery=5,
    markersize=10,
    label="test set",
)
ax1.plot(
    train_accuracy,
    lw=3,
    marker="^",
    markevery=5,
    markersize=10,
    label="train set (stochastic estimate)",
)
ax1.set_ylabel("Accuracy", fontsize=20)
ax1.grid()
ax1.set_xlabel("Epochs", fontsize=20)
ax1.set_xlim((0, MAX_EPOCHS))
ax1.set_ylim((0, 1))

ax2.plot(
    test_losses, lw=3, marker="s", markevery=5, markersize=10, label="test set"
)
ax2.plot(
    train_losses,
    lw=3,
    marker="^",
    markevery=5,
    markersize=10,
    label="train set (stochastic estimate)",
)

ax2.set_ylabel("Loss", fontsize=20)
ax2.grid()
ax2.set_xlabel("Epochs", fontsize=20)
ax2.set_xlim((0, MAX_EPOCHS))

ax1.legend(
    frameon=False, fontsize=20, ncol=2, loc=2, bbox_to_anchor=(0.3, -0.1)
)

ax2.set_yscale("log")

plt.show()
```

```{code-cell} ipython3
# Finally, let's print the test set accuracy
print("Final accuracy on test set: ", test_accuracy[-1])
```
