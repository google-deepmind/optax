# Adversarial training


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/optax/blob/main/examples/adversarial_training.ipynb)

The following code trains a convolutional neural network (CNN) to be robust
with respect to the projected gradient descent (PGD) method.

The Projected Gradient Descent Method (PGD) is a simple yet effective method to
generate adversarial images. At each iteration, it adds a small perturbation
in the direction of the sign of the gradient with respect to the input followed
by a projection onto the infinity ball. The gradient sign ensures this
perturbation locally maximizes the objective, while the projection ensures this
perturbation stays on the boundary of the infinity ball.

## References

  Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining
  and harnessing adversarial examples.", https://arxiv.org/abs/1412.6572

  Madry, Aleksander, et al. "Towards deep learning models resistant to
  adversarial attacks.", https://arxiv.org/abs/1706.06083


To run the colab locally you need install the
`tensorflow-datasets`, `grain` packages via `pip`.

```{code-cell} ipython3
import datetime

import jax
from jax import numpy as jnp
from flax import nnx

import optax
from optax.losses import softmax_cross_entropy_with_integer_labels
from optax.tree_utils import tree_norm

from matplotlib import pyplot as plt
plt.rcParams.update({"font.size": 22})

import tensorflow_datasets as tfds
import grain

# Show on which platform JAX is running.
print("JAX running on", jax.devices()[0].platform.upper())
```

```{code-cell} ipython3
# @markdown Total number of epochs to train for:
EPOCHS = 10  # @param{type:"integer"}
# @markdown Number of samples for each batch in the training set:
TRAIN_BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown Number of samples for each batch in the test set:
TEST_BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown Learning rate for the optimizer:
LEARNING_RATE = 0.001  # @param{type:"number"}
# @markdown The dataset to use.
DATASET = "mnist"  # @param{type:"string"}
# @markdown The amount of L2 regularization to use:
L2_REG = 0.0001  # @param{type:"number"}
# @markdown Adversarial perturbations lie within the infinity-ball of radius epsilon.
EPSILON = 0.01  # @param{type:"number"}
```

```{code-cell} ipython3
class CNN(nnx.Module):
  """A simple CNN model using flax.nnx."""
  def __init__(self, num_classes: int, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(in_features=1, out_features=32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), rngs=rngs)
    self.dense1 = nnx.Linear(in_features=3136, out_features=256, rngs=rngs) # 3136 is for 28x28 input
    self.dense2 = nnx.Linear(in_features=256, out_features=num_classes, rngs=rngs)

  def __call__(self, x):
    x = self.conv1(x)
    x = nnx.relu(x)
    x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = self.conv2(x)
    x = jax.nn.relu(x)
    x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = self.dense1(x)
    x = nnx.relu(x)
    x = self.dense2(x)
    return x
```

```{code-cell} ipython3
train_source, test_source = tfds.data_source("mnist", split=["train", "test"])

train_loader_batched = (
    grain.MapDataset.source(train_source)
    .shuffle(seed=45)
    .map(lambda x: (x["image"], x["label"]))
    .batch(TRAIN_BATCH_SIZE, drop_remainder=True)
)
test_loader_batched = (
    grain.MapDataset.source(test_source)
    .map(lambda x: (x["image"], x["label"]))
    .batch(TEST_BATCH_SIZE, drop_remainder=True)
)

input_shape = (1, *train_source.dataset_info.features["image"].shape)
num_classes = train_source.dataset_info.features["label"].num_classes
iter_per_epoch_train = len(train_loader_batched)
iter_per_epoch_test = len(test_loader_batched)
```

```{code-cell} ipython3
model = CNN(num_classes, rngs=nnx.Rngs(0))

@nnx.jit
def accuracy(model, data):
  inputs, labels = data
  logits = model(inputs)
  return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


@nnx.jit
def loss_fun(model, l2reg, data):
  """Compute the loss of the network."""
  inputs, labels = data
  x = inputs.astype(jnp.float32)
  logits = model(x)
  sqnorm = tree_norm(nnx.state(model, nnx.Param), squared=True)
  loss_value = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
  return loss_value + 0.5 * l2reg * sqnorm

@jax.jit
def pgd_attack(image, label, params, epsilon=0.1, maxiter=10):
  """PGD attack on the L-infinity ball with radius epsilon.

  Args:
    image: array-like, input data for the CNN
    label: integer, class label corresponding to image
    params: tree, parameters of the model to attack
    epsilon: float, radius of the L-infinity ball.
    maxiter: int, number of iterations of this algorithm.

  Returns:
    perturbed_image: Adversarial image on the boundary of the L-infinity ball
      of radius epsilon and centered at image.

  Notes:
    PGD attack is described in (Madry et al. 2017),
    https://arxiv.org/pdf/1706.06083.pdf
  """
  image_perturbation = jnp.zeros_like(image)
  def adversarial_loss(model, perturbation):
    return loss_fun(model, 0, (image + perturbation, label))

  grad_adversarial = nnx.grad(adversarial_loss, argnums=1)
  for _ in range(maxiter):
    # compute gradient of the loss wrt to the image perturbation
    sign_grad = jnp.sign(grad_adversarial(model, image_perturbation))

    # heuristic step-size 2 eps / maxiter
    image_perturbation += (2 * epsilon / maxiter) * sign_grad
    # projection step onto the L-infinity ball centered at image
    image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

  # clip the image to ensure pixels are between 0 and 1
  return jnp.clip(image + image_perturbation, 0, 1)
```

```{code-cell} ipython3
def dataset_stats(model, data_loader, iter_per_epoch):
  """Computes accuracy on clean and adversarial images."""
  adversarial_accuracy = 0.
  clean_accuracy = 0.
  for batch in data_loader:
    images, labels = batch
    images = images.astype(jnp.float32) / 255
    clean_accuracy += jnp.mean(accuracy(model, (images, labels))) / iter_per_epoch
    adversarial_images = pgd_attack(images, labels, model, epsilon=EPSILON)
    adversarial_accuracy += jnp.mean(accuracy(model, (adversarial_images, labels))) / iter_per_epoch
  return {"adversarial accuracy": adversarial_accuracy, "accuracy": clean_accuracy}

@nnx.jit
def train_step(model, optimizer, batch):
  images, labels = batch
  # convert images to float as attack requires to take gradients wrt to them
  images = images.astype(jnp.float32) / 255
  adversarial_images_train = pgd_attack(images, labels, model, epsilon=EPSILON)
  # train on adversarial images
  loss_grad_fun = nnx.value_and_grad(loss_fun)
  loss, grads = loss_grad_fun(model, L2_REG, (adversarial_images_train, labels))
  optimizer.update(model, grads)
```

```{code-cell} ipython3
# Initialize parameters.
model = CNN(num_classes, rngs=nnx.Rngs(0))

# Initialize the optimizer using nnx.
optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

start = datetime.datetime.now().replace(microsecond=0)

accuracy_train = []
accuracy_test = []
adversarial_accuracy_train = []
adversarial_accuracy_test = []
for epoch in range(EPOCHS):
  for train_batch in train_loader_batched:
    train_step(model, optimizer, train_batch)

  # compute train set accuracy, both on clean and adversarial images
  train_stats = dataset_stats(model, train_loader_batched, iter_per_epoch_train)
  accuracy_train.append(train_stats["accuracy"])
  adversarial_accuracy_train.append(train_stats["adversarial accuracy"])

  # compute test set accuracy, both on clean and adversarial images
  test_stats = dataset_stats(model, test_loader_batched, iter_per_epoch_test)
  accuracy_test.append(test_stats["accuracy"])
  adversarial_accuracy_test.append(test_stats["adversarial accuracy"])

  time_elapsed = (datetime.datetime.now().replace(microsecond=0) - start)
  print(f"Epoch {epoch} out of {EPOCHS}")
  print(f"Accuracy on train set: {accuracy_train[-1]:.3f}")
  print(f"Accuracy on test set: {accuracy_test[-1]:.3f}")
  print(f"Adversarial accuracy on train set: {adversarial_accuracy_train[-1]:.3f}")
  print(f"Adversarial accuracy on test set: {adversarial_accuracy_test[-1]:.3f}")
  print(f"Time elapsed: {time_elapsed}\n")
```

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

plt.suptitle("Adversarial training on " + f"{DATASET}".upper())
axes[0].plot(
    accuracy_train, lw=3, label="train set.", marker="<", markersize=10
)
axes[0].plot(accuracy_test, lw=3, label="test set.", marker="d", markersize=10)
axes[0].grid()
axes[0].set_ylabel("accuracy on clean images")

axes[1].plot(
    adversarial_accuracy_train,
    lw=3,
    label="adversarial accuracy on train set.",
    marker="^",
    markersize=10,
)
axes[1].plot(
    adversarial_accuracy_test,
    lw=3,
    label="adversarial accuracy on test set.",
    marker=">",
    markersize=10,
)
axes[1].grid()
axes[0].legend(
    frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.8, -0.1)
)
axes[0].set_xlabel("epochs")
axes[1].set_ylabel("accuracy on adversarial images")
plt.subplots_adjust(wspace=0.5)


plt.show()
```

Find a test set image that is correctly classified but not its adversarial perturbation

```{code-cell} ipython3
def find_adversarial_imgs(model, loader_batched):
  """Finds a test set image that is correctly classified but not its adversarial perturbation."""
  for batch in loader_batched:
    images, labels = batch
    images = images.astype(jnp.float32) / 255
    logits = model(images)
    labels_clean = jnp.argmax(logits, axis=-1)

    adversarial_images = pgd_attack(images, labels, model, epsilon=EPSILON)
    labels_adversarial = jnp.argmax(
        model(adversarial_images), axis=-1
    )
    idx_misclassified = jnp.where(labels_clean != labels_adversarial)[0]
    for j in idx_misclassified:
      clean_image = images[j]
      prediction_clean = labels_clean[j]
      if prediction_clean != labels[j]:
        # the clean image predicts the wrong label, skip
        continue
      adversarial_image = adversarial_images[j]
      adversarial_prediction = labels_adversarial[j]
      # we found our image
      return (
          clean_image,
          prediction_clean,
          adversarial_image,
          adversarial_prediction,
      )

  raise ValueError("No mismatch between clean and adversarial prediction found")


img_clean, pred_clean, img_adversarial, prediction_adversarial = (
    find_adversarial_imgs(model, test_loader_batched)
)
```

```{code-cell} ipython3
_, axes = plt.subplots(nrows=1, ncols=3, figsize=(6 * 3, 6))

axes[0].set_title("Clean image \n Prediction %s" % int(pred_clean))
axes[0].imshow(img_clean, cmap=plt.cm.get_cmap("Greys"), vmax=1, vmin=0)
axes[1].set_title("Adversarial image \n Prediction %s" % prediction_adversarial)
axes[1].imshow(img_adversarial, cmap=plt.cm.get_cmap("Greys"), vmax=1, vmin=0)
axes[2].set_title(r"|Adversarial - clean| $\times$ %.0f" % (1 / EPSILON))
axes[2].imshow(
    jnp.abs(img_clean - img_adversarial) / EPSILON,
    cmap=plt.cm.get_cmap("Greys"),
    vmax=1,
    vmin=0,
)
for i in range(3):
  axes[i].set_xticks(())
  axes[i].set_yticks(())
plt.show()
```

```{code-cell} ipython3

```
