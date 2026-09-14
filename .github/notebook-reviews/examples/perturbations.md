---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Perturbed optimizers


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/perturbations.ipynb)
<!-- #endregion -->

<!-- #region id="l98gJVtFFJPB" -->
We review in this notebook a universal method to transform any function $f$ mapping a pytree to another pytree to a differentiable approximation $f_\varepsilon$, using pertutbations following the method of [Berthet et al. (2020)](https://arxiv.org/abs/2002.08676).

For a random $Z$ drawn from a distribution with continuous positive distribution $\mu$ and a function $f: X \to Y$, its perturbed approximation defined for any $x \in X$ by

$$f_\varepsilon(x) = \mathbf{E}[f (x + \varepsilon Z )]\, .$$

We illustrate here on some examples, including the case of an optimizer function $y^*$ over $C$ defined for any cost $\theta \in \mathbb{R}^d$ by

$$y^*(\theta) = \mathop{\mathrm{arg\,max}}_{y \in C} \langle y, \theta \rangle\, .$$

In this case, the perturbed optimizer is given by

$$y_\varepsilon^*(\theta) = \mathbf{E}[\mathop{\mathrm{arg\,max}}_{y\in C} \langle y, \theta + \varepsilon Z \rangle]\, .$$
<!-- #endregion -->

```python id="S6tLyyy9VCEw"
import jax
import jax.numpy as jnp
import operator
from jax import tree_util as jtu

import optax.tree
from optax import perturbations
```

<!-- #region id="EmYn_jNUFfw2" -->
# Argmax one-hot
<!-- #endregion -->

<!-- #region id="BmHzgCn7FJPC" -->
We consider an optimizer, such as the following `argmax_one_hot` function. It transforms a real-valued vector into a binary vector with a 1 in the coefficient with largest magnitude and 0 elsewhere. It corresponds to $y^*$ for $C$ being the unit simplex. We run it on an example input `values`.
<!-- #endregion -->

<!-- #region id="84N-wAJ8GDK2" -->
## One-hot function
<!-- #endregion -->

```python id="kMZnzhX4FjGj"
def argmax_one_hot(x, axis=-1):
  return jax.nn.one_hot(jnp.argmax(x, axis=axis), x.shape[axis])
```

```python id="iynCk8734Wiz"
values = jnp.array([-0.6, 1.9, -0.2, 1.1, -1.0])

one_hot_vec = argmax_one_hot(values)
print(one_hot_vec)
```

<!-- #region id="6rbNt-6zGb-J" -->
## One-hot with pertubations
<!-- #endregion -->

<!-- #region id="5lvIhCV1FJPD" -->
Our implementation transforms the `argmax_one_hot` function into a perturbed one that we call `pert_one_hot`. In this case we use Gumbel noise for the perturbation.
<!-- #endregion -->

```python id="7hQz6zuPwkpZ"
N_SAMPLES = 100
SIGMA = 0.5
GUMBEL = perturbations.Gumbel()

rng = jax.random.PRNGKey(1)
pert_one_hot = perturbations.make_perturbed_fun(fun=argmax_one_hot,
                                                num_samples=N_SAMPLES,
                                                sigma=SIGMA,
                                                noise=GUMBEL)
```

<!-- #region id="I2VBjnSUFJPD" -->
In this particular case, it is equal to the usual [softmax function](https://en.wikipedia.org/wiki/Softmax_function). This is not always true, in general there is no closed form for $y_\varepsilon^*$
<!-- #endregion -->

```python id="f2gDpghJYZ33"
rngs = jax.random.split(rng, 2)

rng = rngs[0]

pert_argmax = pert_one_hot(rng, values)
print(f'computation with {N_SAMPLES} samples, sigma = {SIGMA}')
print(f'perturbed argmax = {pert_argmax}')
jax.nn.softmax(values/SIGMA)
soft_max = jax.nn.softmax(values/SIGMA)
print(f'softmax = {soft_max}')
print(f'square norm of softmax = {jnp.linalg.norm(soft_max):.2e}')
print(f'square norm of difference = {jnp.linalg.norm(pert_argmax - soft_max):.2e}')
```

<!-- #region id="2U7rhtEAGpMV" -->
## Gradients for one-hot with perturbations
<!-- #endregion -->

<!-- #region id="ldxsWvLmFJPD" -->
The perturbed optimizer $y_\varepsilon^*$ is differentiable, and its gradient can be computed with stochastic estimation automatically, using `jax.grad`.

We create a scalar loss `loss_simplex` of the perturbed optimizer $y^*_\varepsilon$

$$\ell_\text{simplex}(y_{\text{true}} = y_\varepsilon^*; y_{\text{true}})$$  

For `values` equal to a vector $\theta$, we can compute gradients of

$$\ell(\theta) = \ell_\text{simplex}(y_\varepsilon^*(\theta); y_{\text{true}})$$
with respect to `values`, automatically.
<!-- #endregion -->

```python id="7H1LD4QhGtFI"
# Example loss function

def loss_simplex(values, rng):
  n = values.shape[0]
  v_true = jnp.arange(n) + 2
  y_true = v_true / jnp.sum(v_true)
  y_pred = pert_one_hot(rng, values)
  return jnp.sum((y_true - y_pred) ** 2)

loss_simplex(values, rngs[1])
```

<!-- #region id="CM2poXb4FJPD" -->
We can compute the gradient of $\ell$ directly

$$\nabla_\theta \ell(\theta) = \partial_\theta y^*_\varepsilon(\theta) \cdot \nabla_1 \ell_{\text{simplex}}(y^*_\varepsilon(\theta); y_{\text{true}})$$

The computation of the jacobian $\partial_\theta y^*_\varepsilon(\theta)$ is implemented automatically, using an estimation method given by [Berthet et al. (2020)](https://arxiv.org/abs/2002.08676), [Prop. 3.1].
<!-- #endregion -->

```python id="tjQatCE3GtFJ"
# Gradient of the loss w.r.t input values

gradient = jax.grad(loss_simplex)(values, rngs[1])
print(gradient)
```

<!-- #region id="Eh2Qt97AFJPD" -->
We illustrate the use of this method by running 200 steps of gradient descent on $\theta_t$ so that it minimizes this loss.
<!-- #endregion -->

```python id="MuNE2RX0GtFJ"
# Doing 200 steps of gradient descent on the values to have the desired ranks

steps = 200
values_t = values
eta = 0.5

grad_func = jax.jit(jax.grad(loss_simplex))

for t in range(steps):
  rngs = jax.random.split(rngs[1], 2)
  values_t = values_t - eta * grad_func(values_t, rngs[1])
```

```python id="29TWHiH0GtFJ"
rngs = jax.random.split(rngs[1], 2)

n = values.shape[0]
v_true = jnp.arange(n) + 2
y_true = v_true / jnp.sum(v_true)

print(f'initial values = {values}')
print(f'initial one-hot = {argmax_one_hot(values)}')
print(f'initial diff. one-hot = {pert_one_hot(rngs[0], values)}')
print()
print(f'values after GD = {values_t}')
print(f'ranks after GD = {argmax_one_hot(values_t)}')
print(f'diff. one-hot after GD = {pert_one_hot(rngs[1], values_t)}')
print(f'target diff. one-hot = {y_true}')
```

<!-- #region id="4Vyh_a1bZT-s" -->
# Differentiable ranking
<!-- #endregion -->

<!-- #region id="QmVAjbJxFzUA" -->
## Ranking function
<!-- #endregion -->

<!-- #region id="gyapGu77FJPE" -->
We consider an optimizer, such as the following `ranking` function. It transforms a real-valued vector of size $n$ into a vector with coefficients being a permutation of $\{0,\ldots, n-1\}$ corresponding to the order of the coefficients of the original vector. It corresponds to $y^*$ for $C$ being the permutahedron. We run it on an example input `values`.
<!-- #endregion -->

```python id="-NKbR6TlZUTG"
# Function outputting a vector of ranks

def ranking(values):
  return jnp.argsort(jnp.argsort(values))
```

```python id="iU69uMAoZncY"
# Example on random values

n = 6

rng = jax.random.PRNGKey(0)
values = jax.random.normal(rng, (n,))

print(f'values = {values}')
print(f'ranking = {ranking(values)}')
```

<!-- #region id="5j1Vgfz_bb9u" -->
## Ranking with perturbations
<!-- #endregion -->

<!-- #region id="eu2wfbNuFJPE" -->
As above, our implementation transforms this function into a perturbed one that we call `pert_ranking`. In this case we use Gumbel noise for the perturbation.
<!-- #endregion -->

```python id="Equ3_gDPbf5n"
N_SAMPLES = 100
SIGMA = 0.2
GUMBEL = perturbations.Gumbel()

pert_ranking = perturbations.make_perturbed_fun(ranking,
                                                num_samples=N_SAMPLES,
                                                sigma=SIGMA,
                                                noise=GUMBEL)
```

```python id="vMj-Dnudby_a"
# Expectation of the perturbed ranks on these values

rngs = jax.random.split(rng, 2)

diff_ranks = pert_ranking(rngs[0], values)
print(f'values = {values}')

print(f'diff_ranks = {diff_ranks}')
```

<!-- #region id="aH6Ew85koQvU" -->
## Gradients for ranking with perturbations
<!-- #endregion -->

<!-- #region id="UDZOEt18FJPE" -->
As above, the perturbed optimizer $y_\varepsilon^*$ is differentiable, and its gradient can be computed with stochastic estimation automatically, using `jax.grad`.

We showcase this on a loss of $y_\varepsilon(\theta)$ that can be directly differentiated w.r.t. the `values` equal to $\theta$.
<!-- #endregion -->

```python id="O-T8y6N8cHzF"
# Example loss function

def loss_example(values, rng):
  n = values.shape[0]
  y_true = ranking(jnp.arange(n))
  y_pred = pert_ranking(rng, values)
  return jnp.sum((y_true - y_pred) ** 2)

print(loss_example(values, rngs[1]))
```

```python id="v7nzNwP-e68q"
# Gradient of the objective w.r.t input values

gradient = jax.grad(loss_example)(values, rngs[1])
print(gradient)
```

<!-- #region id="aC7IzKADFJPE" -->
As above, we showcase this example on gradient descent to minimize this loss.
<!-- #endregion -->

```python id="0UObBP3QfCqq"
steps = 20
values_t = values
eta = 0.1

grad_func = jax.jit(jax.grad(loss_example))

for t in range(steps):
  rngs = jax.random.split(rngs[1], 2)
  values_t = values_t - eta * grad_func(values_t, rngs[1])
```

```python id="p4iNxMoQmZRa"
rngs = jax.random.split(rngs[1], 2)

y_true = ranking(jnp.arange(n))

print(f'initial values = {values}')
print(f'initial ranks = {ranking(values)}')
print(f'initial diff. ranks = {pert_ranking(rngs[0], values)}')
print()
print(f'values after GD = {values_t}')
print(f'ranks after GD = {ranking(values_t)}')
print(f'diff. ranks after GD = {pert_ranking(rngs[1], values_t)}')
print(f'target diff. ranks = {y_true}')
```

<!-- #region id="P537S89ZlDQR" -->
# General input / outputs (Pytrees)
<!-- #endregion -->

<!-- #region id="V62NPuUvSHk8" -->
This method can be applied to any function taking pytrees as input and output in the forward mode, and can also be used to compute derivatives, as illustrated below
<!-- #endregion -->

```python id="0Bz35ZWQpeB7"
tree_a = (jnp.array((0.1, 0.4, 0.5)),
          {'k1': jnp.array((0.1, 0.2)),
           'k2': jnp.array((0.1, 0.1))},
          jnp.array((0.4, 0.3, 0.2, 0.1)))
```

<!-- #region id="UxcczrOZhCZJ" -->
## Tree argmax
<!-- #endregion -->

<!-- #region id="hUQqWJYfgrag" -->
This piecewise constant function applies the argmax to every leaf array of the pytree
<!-- #endregion -->

```python id="szID1S5Jg_LL"
argmax_tree = lambda x: jax.tree.map(argmax_one_hot, x)
```

```python id="KPfzjdSJxP4G"
argmax_tree(tree_a)
```

<!-- #region id="hOmGdkW0g2-6" -->
The perturbed approximation applies a perturbed softmax
<!-- #endregion -->

```python id="oKuD_cElxSDd"
N_SAMPLES = 100
sigma = 1.0

pert_argmax_fun = perturbations.make_perturbed_fun(argmax_tree,
                                                   num_samples=N_SAMPLES,
                                                   sigma=SIGMA)
```

```python id="_fnpfpVBxYSQ"
pert_argmax_fun(rng, tree_a)
```

<!-- #region id="MQOh_iZmhLVS" -->
## Scalar loss
<!-- #endregion -->

```python id="zW0U0DW1xbAV"
def pert_loss(inputs, rng):
  pert_softmax = pert_argmax_fun(rng, inputs)
  argmax = argmax_tree(inputs)
  diffs = jax.tree.map(lambda x, y: jnp.sum((x - y) ** 2 / 4), argmax, pert_softmax)
  return jax.tree.reduce(operator.add, diffs)
```

```python id="bWjXKeMSxodX"
init_loss = pert_loss(tree_a, rng)

print(f'initial loss value = {init_loss:.3f}')
```

<!-- #region id="eXobsx6bhRb8" -->
## Gradient computation
<!-- #endregion -->

<!-- #region id="kydUMAachVgp" -->
The gradient of the scalar loss can be evaluated
<!-- #endregion -->

```python id="vryBVzPsxqlI"
grad = jax.grad(pert_loss)(tree_a, rng)

print('Gradient of the scalar loss')
print()
grad
```

<!-- #region id="SpbnFapshcaM" -->
A small step in the gradient direction reduces the value
<!-- #endregion -->

```python id="EkwIh76L1Azl"
eta = 1e-1

loss_step = pert_loss(optax.tree.add_scale(tree_a, -eta, grad), rng)

print(f'initial loss value = {init_loss:.3f}')
print(f'loss after gradient step = {loss_step:.3f}')
```
