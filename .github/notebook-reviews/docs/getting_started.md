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

<!-- #region id="EXQz7Vp8ehqb" -->
# 🚀 Getting started

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/docs/getting_started.ipynb)

Optax is a simple optimization library for [JAX](https://jax.readthedocs.io/). The main object is the {py:class}`GradientTransformation <optax.GradientTransformation>`, which can be chained with other transformations to obtain the final update operation and the optimizer state. Optax also contains some simple loss functions and utilities to help you write the full optimization steps. This notebook walks you through a few examples on how to use Optax.
<!-- #endregion -->

<!-- #region id="vEIU3POrGiE5" -->
## Example: Fitting a Linear Model

Begin by importing the necessary packages:
<!-- #endregion -->

```python id="Jr7_e_ZJ_hky"
import jax.numpy as jnp
import jax
import optax
import functools
```

<!-- #region id="n7kMS9kyM8vM" -->
In this example, we begin by setting up a simple linear model and a loss function. You can use any other library, such as [haiku](https://github.com/deepmind/dm-haiku) or [Flax](https://github.com/google/flax) to construct your networks. Here, we keep it simple and write it ourselves. The loss function (L2 Loss) comes from Optax's {doc}`losses <api/losses>` via {py:class}`l2_loss <optax.l2_loss>`.
<!-- #endregion -->

```python id="0-8XwoQF_AO2"
@functools.partial(jax.vmap, in_axes=(None, 0))
def network(params, x):
  return jnp.dot(params, x)

def compute_loss(params, x, y):
  y_pred = network(params, x)
  loss = jnp.mean(optax.l2_loss(y_pred, y))
  return loss
```

<!-- #region id="EZviuSmuNFsC" -->
Here we generate data under a known linear model (with `target_params=0.5`):
<!-- #endregion -->

```python id="H-_pwBx6_keL"
key = jax.random.PRNGKey(42)
target_params = 0.5

# Generate some data.
xs = jax.random.normal(key, (16, 2))
ys = jnp.sum(xs * target_params, axis=-1)
```

<!-- #region id="Td4Lp3qDNsL3" -->
### Basic usage of Optax

Optax contains implementations of {doc}`many popular optimizers <api/optimizers>` that can be used very simply. For example, the gradient transform for the Adam optimizer is available at {py:class}`optax.adam`. For now, let's start by calling the {py:class}`GradientTransformation <optax.GradientTransformation>` object for Adam the `optimizer`. We then initialize the optimizer state using the `init` function and `params` of the network.
<!-- #endregion -->

```python id="rsLXLb5wBeY2"
start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)

# Initialize parameters of the model + optimizer.
params = jnp.array([0.0, 0.0])
opt_state = optimizer.init(params)
```

<!-- #region id="CpAvP1WSnsyM" -->
Next we write the update loop. The {py:class}`GradientTransformation <optax.GradientTransformation>` object contains an `update` function that takes in the current optimizer state and gradients and returns the `updates` that need to be applied to the parameters: `updates, new_opt_state = optimizer.update(grads, opt_state)`.

Optax comes with a few simple {doc}`update rules <api/apply_updates>` that apply the updates from the gradient transforms to the current parameters to return new ones: `new_params = optax.apply_updates(params, updates)`.
<!-- #endregion -->

```python id="TNkhz_nrB2lx"
# A simple update loop.
for _ in range(1000):
  grads = jax.grad(compute_loss)(params, xs, ys)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

assert jnp.allclose(params, target_params), \
'Optimization should retrieve the target params used to generate the data.'
```

<!-- #region id="XXEz3j7wPZUH" -->
### Custom optimizers

Optax makes it easy to create custom optimizers by {py:class}`chain <optax.chain>`ing gradient transforms. For example, this creates an optimizer based on Adam. Note that the scaling is `-learning_rate` which is an important detail since {py:class}`apply_updates <optax.apply_updates>` is additive.
<!-- #endregion -->

```python id="KQNI2P3YEEgP"
# Exponential decay of the learning rate.
scheduler = optax.exponential_decay(
    init_value=start_learning_rate,
    transition_steps=1000,
    decay_rate=0.99)

# Combining gradient transforms using `optax.chain`.
gradient_transform = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
    optax.scale_by_adam(),  # Use the updates from adam.
    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0)
)
```

```python id="XGUrLKxAEO3j"
# Initialize parameters of the model + optimizer.
params = jnp.array([0.0, 0.0])  # Recall target_params=0.5.
opt_state = gradient_transform.init(params)

# A simple update loop.
for _ in range(1000):
  grads = jax.grad(compute_loss)(params, xs, ys)
  updates, opt_state = gradient_transform.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

assert jnp.allclose(params, target_params), \
'Optimization should retrieve the target params used to generate the data.'
```

<!-- #region id="pIxKL7WsXFl8" -->
### Advanced usage of Optax
<!-- #endregion -->

<!-- #region id="nCtNiVTsZVt2" -->
#### Modifying hyperparameters of optimizers in a schedule.

In some scenarios, changing the hyperparameters (other than the learning rate) of an optimizer can be useful to ensure training reliability. We can do this easily by using {py:class}`inject_hyperparams <optax.inject_hyperparams>`. For example, this piece of code decays the `max_norm` of the {py:class}`clip_by_global_norm <optax.clip_by_global_norm>` gradient transform as training progresses:



<!-- #endregion -->

```python id="NR9Flsj7ZdpC"
decaying_global_norm_tx = optax.inject_hyperparams(optax.clip_by_global_norm)(
    max_norm=optax.linear_schedule(1.0, 0.0, transition_steps=99))

opt_state = decaying_global_norm_tx.init(None)
assert opt_state.hyperparams['max_norm'] == 1.0, 'Max norm should start at 1.0'

for _ in range(100):
  _, opt_state = decaying_global_norm_tx.update(None, opt_state)

assert opt_state.hyperparams['max_norm'] == 0.0, 'Max norm should end at 0.0'
```

<!-- #region id="tKcocLxEyYf2" -->
## Example: Fitting a MLP

Let's use Optax to fit a parametrized function. We will consider the problem of learning to identify when a value is odd or even.

We will begin by creating a dataset that consists of batches of random 8 bit integers (represented using their binary representation), with each value labelled as "odd" or "even" using 1-hot encoding (i.e. `[1, 0]` means odd `[0, 1]` means even).

<!-- #endregion -->

```python id="Gg6zyMBqydty"
import optax
import jax.numpy as jnp
import jax
import numpy as np

BATCH_SIZE = 5
NUM_TRAIN_STEPS = 1_000
RAW_TRAINING_DATA = np.random.randint(255, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 1))

TRAINING_DATA = np.unpackbits(RAW_TRAINING_DATA.astype(np.uint8), axis=-1)
LABELS = jax.nn.one_hot(RAW_TRAINING_DATA % 2, 2).astype(jnp.float32).reshape(NUM_TRAIN_STEPS, BATCH_SIZE, 2)
```

<!-- #region id="nV79rjQK8tvC" -->
We may now define a parametrized function using JAX. This will allow us to efficiently compute gradients.

There are a number of libraries that provide common building blocks for parametrized functions (such as flax and haiku). For this case though, we shall implement our function from scratch.

Our function will be a 1-layer MLP (multi-layer perceptron) with a single hidden layer, and a single output layer. We initialize all parameters using a standard Gaussian {math}`\mathcal{N}(0,1)` distribution.
<!-- #endregion -->

```python id="Syp9LJ338h9-"
initial_params = {
    'hidden': jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),
    'output': jax.random.normal(shape=[32, 2], key=jax.random.PRNGKey(1)),
}


def net(x: jnp.ndarray, params: optax.Params) -> jnp.ndarray:
  x = jnp.dot(x, params['hidden'])
  x = jax.nn.relu(x)
  x = jnp.dot(x, params['output'])
  return x


def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  y_hat = net(batch, params)

  # optax also provides a number of common loss functions.
  loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)

  return loss_value.mean()
```

<!-- #region id="2LVHrJyH9vDe" -->
We will use {py:class}`optax.adam` to compute the parameter updates from their gradients on each optimizer step.

Note that since Optax optimizers are implemented using pure functions, we will need to also keep track of the optimizer state. For the Adam optimizer, this state will contain the momentum values.
<!-- #endregion -->

```python id="JsbPBTF09FGY"
def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
  opt_state = optimizer.init(params)

  @jax.jit
  def step(params, opt_state, batch, labels):
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

  for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
    params, opt_state, loss_value = step(params, opt_state, batch, labels)
    if i % 100 == 0:
      print(f'step {i}, loss: {loss_value}')

  return params, opt_state

# Finally, we can fit our parametrized function using the Adam optimizer
# provided by optax.
optimizer = optax.adam(learning_rate=1e-2)
_ = fit(initial_params, optimizer)
```

<!-- #region id="kTaBLYL8_Ppz" -->
We see that our loss appears to have converged, which should indicate that we have successfully found better parameters for our network.
<!-- #endregion -->

<!-- #region id="qT_Uaei5Dv_3" -->
### Weight Decay, Schedules and Clipping

Many research models make use of techniques such as learning rate scheduling, and gradient clipping. These may be achieved by chaining together gradient transformations such as {py:class}`optax.adam` and {py:class}`optax.clip`.

In the following, we will use Adam with weight decay ({py:class}`optax.adamw`), a cosine learning rate schedule (with warmup) and also gradient clipping.
<!-- #endregion -->

```python id="SZegYQajDtLi"
schedule = optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=1.0,
  warmup_steps=50,
  decay_steps=1_000,
  end_value=0.0,
)

optimizer = optax.chain(
  optax.clip(1.0),
  optax.adamw(learning_rate=schedule),
)

_ = fit(initial_params, optimizer)
```

<!-- #region id="bluOz9x8HUr5" -->
### Accessing learning rate

Optimizer states only contain the information needed to be stored for the next iteration. The value of the learning rate does not need to be stored: we only need the current count of the optimizer to output the next learning rate in the schedule.

The optimizer may still be defined in such a way that the learning rate is stored in the state by using the `optax.inject_hyperparams`.
<!-- #endregion -->

```python id="6P-cM94kNcdt"
optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=schedule)

params = initial_params
state = optimizer.init(params)
print('initial learning rate:', state.hyperparams['learning_rate'])

_, state = fit(initial_params, optimizer)

print('final learning rate:', state.hyperparams['learning_rate'])
```

<!-- #region id="qf53Y6mT1Vwl" -->
## Components

We refer to the {doc}`docs <index>` for a detailed list of available Optax components. Here, we highlight the main categories of building blocks provided by Optax.
<!-- #endregion -->

<!-- #region id="WZFpEKi82TGx" -->
### Gradient Transformations ([transform.py](https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py))

One of the key building blocks of Optax is a {py:class}`GradientTransformation <optax.GradientTransformation>`. Each transformation is defined by two functions:

`state = init(params)`

`grads, state = update(grads, state, params=None)`
<!-- #endregion -->

<!-- #region id="n6SsC9lNGiE-" -->
The `init` function initializes a (possibly empty) set of statistics (aka state) and the `update` function transforms a candidate gradient given some statistics, and (optionally) the current value of the parameters.

For example:
<!-- #endregion -->

```python id="_yCQbSCc2KhJ"
tx = optax.scale_by_rms()
state = tx.init(params)  # init stats
grads = jax.grad(loss)(params, TRAINING_DATA, LABELS)
updates, state = tx.update(grads, state, params)  # transform & update stats.
```

<!-- #region id="TyxJmbBq2xT6" -->
### Composing Gradient Transformations ([combine.py](https://github.com/google-deepmind/optax/blob/main/optax/_src/combine.py))

The fact that transformations take candidate gradients as input and return processed gradients as output (in contrast to returning the updated parameters) is critical to allow to combine arbitrary transformations into a custom optimizer / gradient processor, and also allows to combine transformations for different gradients that operate on a shared set of variables.

For instance, {py:class}`chain <optax.chain>` combines them sequentially, and returns a new {py:class}`GradientTransformation <optax.GradientTransformation>` that applies several transformations in sequence.

For example:
<!-- #endregion -->

```python id="TNPC9e7I28m8"
max_norm = 100.
learning_rate = 1e-3

my_optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm),
    optax.scale_by_adam(eps=1e-4),
    optax.scale(-learning_rate))
```

<!-- #region id="JmV92-PI2_pS" -->
### Wrapping Gradient Transformations ([wrappers.py](https://github.com/google-deepmind/optax/blob/main/optax/_src/wrappers.py))

Optax also provides several wrappers that take a {py:class}`GradientTransformation <optax.GradientTransformation>` as input and return a new {py:class}`GradientTransformation <optax.GradientTransformation>` that modifies the behavior of the inner transformation in a specific way.

For instance, the {py:class}`flatten <optax.flatten>` wrapper flattens gradients into a single large vector before applying the inner {py:class}`GradientTransformation <optax.GradientTransformation>`. The transformed updates are then unflattened before being returned to the user. This can be used to reduce the overhead of performing many calculations on lots of small variables, at the cost of increasing memory usage.

For example:
<!-- #endregion -->

```python id="b1TlMbAk3Jbo"
my_optimizer = optax.flatten(optax.adam(learning_rate))
```

<!-- #region id="IUCIMymV3M2n" -->
Other examples of wrappers include accumulating gradients over multiple steps or applying the inner transformation only to specific parameters or at specific steps.
<!-- #endregion -->

<!-- #region id="AGAmqST33PkO" -->
### Schedules ([schedule.py](https://github.com/google-deepmind/optax/blob/main/optax/_src/schedule.py))

Many popular transformations use time-dependent components, e.g. to anneal some hyper-parameter (e.g. the learning rate). Optax provides for this purpose schedules that can be used to decay scalars as a function of a `step` count.

For example, you may use a {py:class}`polynomial_schedule <optax.polynomial_schedule>` (with `power=1`) to decay a hyper-parameter linearly over a number of steps:
<!-- #endregion -->

```python id="Zbr61DLP3ecy"
schedule_fn = optax.polynomial_schedule(
    init_value=1., end_value=0., power=1, transition_steps=5)

for step_count in range(6):
  print(schedule_fn(step_count))  # [1., 0.8, 0.6, 0.4, 0.2, 0.]
```

<!-- #region id="LGt0AzHF3fjR" -->
Schedules can be combined with other transforms as follows.
<!-- #endregion -->

```python id="W9oCb0Kw3igG"
schedule_fn = optax.polynomial_schedule(
    init_value=-learning_rate, end_value=0., power=1, transition_steps=5)
optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm),
    optax.scale_by_adam(eps=1e-4),
    optax.scale_by_schedule(schedule_fn))
```

<!-- #region id="sDSXlRAN_B2F" -->
Schedules can also be used in place of the `learning_rate` argument of a
{py:class}`GradientTransformation <optax.GradientTransformation>` as

<!-- #endregion -->

```python id="zyvlGLDw_BKk"
optimizer = optax.adam(learning_rate=schedule_fn)
```

<!-- #region id="cKHZrM203kx4" -->
### Popular optimizers ([alias.py](https://github.com/google-deepmind/optax/blob/main/optax/_src/alias.py))

In addition to the low-level building blocks, we also provide aliases for popular optimizers built using these components (e.g. RMSProp, Adam, AdamW, etc, ...). These are all still instances of a {py:class}`GradientTransformation <optax.GradientTransformation>`, and can therefore be further combined with any of the individual building blocks.

For example:
<!-- #endregion -->

```python id="Czk49AQz3w1J"
def adamw(learning_rate, b1, b2, eps, weight_decay):
  return optax.chain(
      optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
      optax.scale_and_decay(-learning_rate, weight_decay=weight_decay))
```

<!-- #region id="j0tD_jWC3zar" -->
### Applying updates ([update.py](https://github.com/google-deepmind/optax/blob/main/optax/_src/update.py))

After transforming an update using a {py:class}`GradientTransformation <optax.GradientTransformation>` or any custom manipulation of the update, you will typically apply the update to a set of parameters. This can be done trivially using `jax.tree.map`.

For convenience, we expose an {py:class}`apply_updates <optax.apply_updates>` function to apply updates to parameters. The function just adds the updates and the parameters together, i.e. `jax.tree.map(lambda p, u: p + u, params, updates)`.
<!-- #endregion -->

```python id="YG-TNzYm4CHt"
updates, state = tx.update(grads, state, params)  # transform & update stats.
new_params = optax.apply_updates(params, updates)  # update the parameters.
```

<!-- #region id="eg85y6_s4C2c" -->
Note that separating gradient transformations from the parameter update is critical to support composing a sequence of transformations (e.g. {py:class}`chain <optax.chain>`), as well as combining multiple updates to the same parameters (e.g. in multi-task settings where different tasks need different sets of gradient transformations).
<!-- #endregion -->

<!-- #region id="dJzW0Flw4FP5" -->
### Losses ([loss.py](https://github.com/google-deepmind/optax/tree/main/optax/losses))

Optax provides a number of standard losses used in deep learning, such as {py:class}`l2_loss <optax.l2_loss>`, {py:class}`softmax_cross_entropy <optax.softmax_cross_entropy>`, {py:class}`cosine_distance <optax.cosine_distance>`, etc.
<!-- #endregion -->

```python id="8JCWgHhJ4PMc"
predictions = net(TRAINING_DATA, params)
loss = optax.huber_loss(predictions, LABELS)
```

<!-- #region id="gAlaEpgQ4QyD" -->
The losses accept batches as inputs, however, they perform no reduction across the batch dimension(s). This is trivial to do in JAX, for example:
<!-- #endregion -->

```python id="45svU6Qr4ThD"
avg_loss = jnp.mean(optax.huber_loss(predictions, LABELS))
sum_loss = jnp.sum(optax.huber_loss(predictions, LABELS))
```

<!-- #region id="MepQR-Cr4VaB" -->
### Second Order ([second_order.py](https://github.com/google-deepmind/optax/tree/main/optax/second_order))

Computing the Hessian or Fisher information matrices for neural networks is typically intractable due to the quadratic memory requirements. Solving for the diagonals of these matrices is often a better solution. The library offers functions for computing these diagonals with sub-quadratic memory requirements.
<!-- #endregion -->
