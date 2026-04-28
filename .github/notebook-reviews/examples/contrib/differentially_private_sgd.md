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

<!-- #region id="uJHywE_oL3j2" -->
# Differentially private convolutional neural network on MNIST.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/differentially_private_sgd.ipynb)

A large portion of this code is forked from the differentially private SGD
example in the [JAX repo](
https://github.com/jax-ml/jax/blob/main/examples/differentially_private_sgd.py).

To run the colab locally you need install the
`dp-accounting`, `tensorflow`, `tensorflow-datasets`, packages via `pip`.


[Differentially Private Stochastic Gradient Descent](https://arxiv.org/abs/1607.00133) requires clipping the per-example parameter
gradients, which is non-trivial to implement efficiently for convolutional
neural networks.  The JAX XLA compiler shines in this setting by optimizing the
minibatch-vectorized computation for convolutional architectures. Train time
takes a few seconds per epoch on a commodity GPU.
<!-- #endregion -->

```python id="VaYIiCnjL3j3"
import warnings
import dp_accounting
import jax
import jax.numpy as jnp
from optax import contrib
from optax import losses
import optax
from jax.example_libraries import stax
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Shows on which platform JAX is running.
print("JAX running on", jax.devices()[0].platform.upper())
```

<!-- #region id="t7Dn8L_Uw0Yb" -->
This table contains hyperparameters and the corresponding expected test accuracy.


| DPSGD  | LEARNING_RATE | NOISE_MULTIPLIER | L2_NORM_CLIP | BATCH_SIZE | NUM_EPOCHS | DELTA | FINAL TEST ACCURACY |
| ------ | ------------- | ---------------- | ------------ | ---------- | ---------- | ----- | ------------------- |
| False  | 0.1           | NA               | NA           | 256        | 20         | NA    | ~99%                |
| True   | 0.25          | 1.3              | 1.5          | 256        | 15         | 1e-5  | ~95%                |
| True   | 0.15          | 1.1              | 1.0          | 256        | 60         | 1e-5  | ~96.6%              |
| True   | 0.25          | 0.7              | 1.5          | 256        | 45         | 1e-5  | ~97%                |
<!-- #endregion -->

```python id="jve2h810L3j3"
# Whether to use DP-SGD or vanilla SGD:
DPSGD = True
# Learning rate for the optimizer:
LEARNING_RATE = 0.25
# Noise multiplier for DP-SGD optimizer:
NOISE_MULTIPLIER = 1.3
# L2 norm clip:
L2_NORM_CLIP = 1.5
# Number of samples in each batch:
BATCH_SIZE = 256
# Number of epochs:
NUM_EPOCHS = 15
# Probability of information leakage:
DELTA = 1e-5
```

<!-- #region id="iLGeV4y4DBkL" -->
CIFAR10 and CIFAR100 are composed of 32x32 images with 3 channels (RGB). We'll now load the dataset using `tensorflow_datasets` and display a few of the first samples.
<!-- #endregion -->

```python id="zynvtk4wDBkL"
(train_loader, test_loader), info = tfds.load(
    "mnist", split=["train", "test"], as_supervised=True, with_info=True
)

min_max_rgb = lambda image, label: (tf.cast(image, tf.float32) / 255., label)
train_loader = train_loader.map(min_max_rgb)
test_loader = test_loader.map(min_max_rgb)

train_loader_batched = train_loader.shuffle(
    buffer_size=10_000, reshuffle_each_iteration=True
).batch(BATCH_SIZE, drop_remainder=True)

NUM_EXAMPLES = info.splits["test"].num_examples
test_batch = next(test_loader.batch(NUM_EXAMPLES, drop_remainder=True).as_numpy_iterator())
```

```python id="o6In7oQ-0EhG"
init_random_params, predict = stax.serial(
    stax.Conv(16, (8, 8), padding="SAME", strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding="VALID", strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(10),
)
```

<!-- #region id="j2OUgc6J0Jsl" -->
This function computes the privacy parameter epsilon for the given number of steps and probability of information leakage `DELTA`.
<!-- #endregion -->

```python id="43177TofzuOa"
def compute_epsilon(steps):
  if NUM_EXAMPLES * DELTA > 1.:
    warnings.warn("Your delta might be too high.")
  q = BATCH_SIZE / float(NUM_EXAMPLES)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  accountant = dp_accounting.rdp.RdpAccountant(orders)
  accountant.compose(dp_accounting.PoissonSampledDpEvent(
      q, dp_accounting.GaussianDpEvent(NOISE_MULTIPLIER)), steps)
  return accountant.get_epsilon(DELTA)
```

```python id="W9mPtPvB0D3X"
@jax.jit
def loss_fn(params, batch):
  images, labels = batch
  logits = predict(params, images)
  return losses.softmax_cross_entropy_with_integer_labels(logits, labels).mean(), logits


@jax.jit
def test_step(params, batch):
  images, labels = batch
  logits = predict(params, images)
  loss = losses.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
  accuracy = (logits.argmax(1) == labels).mean()
  return loss, accuracy * 100
```

```python id="vOet-_860ysL"
if DPSGD:
  tx = contrib.dpsgd(
      learning_rate=LEARNING_RATE, l2_norm_clip=L2_NORM_CLIP,
      noise_multiplier=NOISE_MULTIPLIER, seed=1337)
else:
  tx = optax.sgd(learning_rate=LEARNING_RATE)

_, params = init_random_params(jax.random.PRNGKey(1337), (-1, 28, 28, 1))
opt_state = tx.init(params)
```

```python id="b-NmP7g01EdA"
@jax.jit
def train_step(params, opt_state, batch):
  grad_fn = jax.grad(loss_fn, has_aux=True)
  if DPSGD:
    # Inserts a dimension in axis 1 to use jax.vmap over the batch.
    batch = jax.tree.map(lambda x: x[:, None], batch)
    # Uses jax.vmap across the batch to extract per-example gradients.
    grad_fn = jax.vmap(grad_fn, in_axes=(None, 0))

  grads, _ = grad_fn(params, batch)
  updates, new_opt_state = tx.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state
```

```python id="QMl9dnbJ1OtQ"
accuracy, loss, epsilon = [], [], []

for epoch in range(NUM_EPOCHS):
  for batch in train_loader_batched.as_numpy_iterator():
    params, opt_state = train_step(params, opt_state, batch)

  # Evaluates test accuracy.
  test_loss, test_acc = test_step(params, test_batch)
  accuracy.append(test_acc)
  loss.append(test_loss)
  print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, test accuracy: {test_acc}")

  #
  if DPSGD:
    steps = (1 + epoch) * NUM_EXAMPLES // BATCH_SIZE
    eps = compute_epsilon(steps)
    epsilon.append(eps)
```

```python id="9nsV-9_b2qca"
if DPSGD:
  _, axs = plt.subplots(ncols=3, figsize=(9, 3))
else:
  _, axs = plt.subplots(ncols=2, figsize=(6, 3))

axs[0].plot(accuracy)
axs[0].set_title("Test accuracy")
axs[1].plot(loss)
axs[1].set_title("Test loss")

if DPSGD:
  axs[2].plot(epsilon)
  axs[2].set_title("Epsilon")

plt.tight_layout()
```

```python id="1ubOEWod3OPj"
print(f'Final accuracy: {accuracy[-1]}')
```
