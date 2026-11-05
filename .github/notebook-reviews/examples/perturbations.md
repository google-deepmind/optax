<!-- #region -->
# Perturbed optimizers


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/perturbations.ipynb)
<!-- #endregion -->

We review in this notebook a universal method to transform any function $f$ mapping a pytree to another pytree to a differentiable approximation $f_\varepsilon$, using pertutbations following the method of [Berthet et al. (2020)](https://arxiv.org/abs/2002.08676).

For a random $Z$ drawn from a distribution with continuous positive distribution $\mu$ and a function $f: X \to Y$, its perturbed approximation defined for any $x \in X$ by

$$f_\varepsilon(x) = \mathbf{E}[f (x + \varepsilon Z )]\, .$$

We illustrate here on some examples, including the case of an optimizer function $y^*$ over $C$ defined for any cost $\theta \in \mathbb{R}^d$ by

$$y^*(\theta) = \mathop{\mathrm{arg\,max}}_{y \in C} \langle y, \theta \rangle\, .$$

In this case, the perturbed optimizer is given by

$$y_\varepsilon^*(\theta) = \mathbf{E}[\mathop{\mathrm{arg\,max}}_{y\in C} \langle y, \theta + \varepsilon Z \rangle]\, .$$

```python
import jax
import jax.numpy as jnp
from jax import tree_util as jtu

import optax.tree
from optax import perturbations
```

# Argmax one-hot


We consider an optimizer, such as the following `argmax_one_hot` function. It transforms a real-valued vector into a binary vector with a 1 in the coefficient with largest magnitude and 0 elsewhere. It corresponds to $y^*$ for $C$ being the unit simplex. We run it on an example input `values`.


## One-hot function

```python
def argmax_one_hot(x, axis=-1):
  return jax.nn.one_hot(jnp.argmax(x, axis=axis), x.shape[axis])
```

```python
values = jnp.array([-0.6, 1.9, -0.2, 1.1, -1.0])

one_hot_vec = argmax_one_hot(values)
print(one_hot_vec)
```

## One-hot with pertubations


Our implementation transforms the `argmax_one_hot` function into a perturbed one that we call `pert_one_hot`. In this case we use Gumbel noise for the perturbation.

```python
N_SAMPLES = 100
SIGMA = 0.5
GUMBEL = perturbations.Gumbel()

rng = jax.random.PRNGKey(1)
pert_one_hot = perturbations.make_perturbed_fun(fun=argmax_one_hot,
                                                num_samples=N_SAMPLES,
                                                sigma=SIGMA,
                                                noise=GUMBEL)
```

In this particular case, it is equal to the usual [softmax function](https://en.wikipedia.org/wiki/Softmax_function). This is not always true, in general there is no closed form for $y_\varepsilon^*$

```python
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

## Gradients for one-hot with perturbations


The perturbed optimizer $y_\varepsilon^*$ is differentiable, and its gradient can be computed with stochastic estimation automatically, using `jax.grad`.

We create a scalar loss `loss_simplex` of the perturbed optimizer $y^*_\varepsilon$

$$\ell_\text{simplex}(y_{\text{true}} = y_\varepsilon^*; y_{\text{true}})$$  

For `values` equal to a vector $\theta$, we can compute gradients of

$$\ell(\theta) = \ell_\text{simplex}(y_\varepsilon^*(\theta); y_{\text{true}})$$
with respect to `values`, automatically.

```python
# Example loss function

def loss_simplex(values, rng):
  n = values.shape[0]
  v_true = jnp.arange(n) + 2
  y_true = v_true / jnp.sum(v_true)
  y_pred = pert_one_hot(rng, values)
  return jnp.sum((y_true - y_pred) ** 2)

loss_simplex(values, rngs[1])
```

We can compute the gradient of $\ell$ directly

$$\nabla_\theta \ell(\theta) = \partial_\theta y^*_\varepsilon(\theta) \cdot \nabla_1 \ell_{\text{simplex}}(y^*_\varepsilon(\theta); y_{\text{true}})$$

The computation of the jacobian $\partial_\theta y^*_\varepsilon(\theta)$ is implemented automatically, using an estimation method given by [Berthet et al. (2020)](https://arxiv.org/abs/2002.08676), [Prop. 3.1].

```python
# Gradient of the loss w.r.t input values

gradient = jax.grad(loss_simplex)(values, rngs[1])
print(gradient)
```

We illustrate the use of this method by running 200 steps of gradient descent on $\theta_t$ so that it minimizes this loss.

```python
# Doing 200 steps of gradient descent on the values to have the desired ranks

steps = 200
values_t = values
eta = 0.5

grad_func = jax.jit(jax.grad(loss_simplex))

for t in range(steps):
  rngs = jax.random.split(rngs[1], 2)
  values_t = values_t - eta * grad_func(values_t, rngs[1])
```

```python
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

# Differentiable ranking


## Ranking function


We consider an optimizer, such as the following `ranking` function. It transforms a real-valued vector of size $n$ into a vector with coefficients being a permutation of $\{0,\ldots, n-1\}$ corresponding to the order of the coefficients of the original vector. It corresponds to $y^*$ for $C$ being the permutahedron. We run it on an example input `values`.

```python
# Function outputting a vector of ranks

def ranking(values):
  return jnp.argsort(jnp.argsort(values))
```

```python
# Example on random values

n = 6

rng = jax.random.PRNGKey(0)
values = jax.random.normal(rng, (n,))

print(f'values = {values}')
print(f'ranking = {ranking(values)}')
```

## Ranking with perturbations


As above, our implementation transforms this function into a perturbed one that we call `pert_ranking`. In this case we use Gumbel noise for the perturbation.

```python
N_SAMPLES = 100
SIGMA = 0.2
GUMBEL = perturbations.Gumbel()

pert_ranking = perturbations.make_perturbed_fun(ranking,
                                                num_samples=N_SAMPLES,
                                                sigma=SIGMA,
                                                noise=GUMBEL)
```

```python
# Expectation of the perturbed ranks on these values

rngs = jax.random.split(rng, 2)

diff_ranks = pert_ranking(rngs[0], values)
print(f'values = {values}')

print(f'diff_ranks = {diff_ranks}')
```

## Gradients for ranking with perturbations


As above, the perturbed optimizer $y_\varepsilon^*$ is differentiable, and its gradient can be computed with stochastic estimation automatically, using `jax.grad`.

We showcase this on a loss of $y_\varepsilon(\theta)$ that can be directly differentiated w.r.t. the `values` equal to $\theta$.

```python
# Example loss function

def loss_example(values, rng):
  n = values.shape[0]
  y_true = ranking(jnp.arange(n))
  y_pred = pert_ranking(rng, values)
  return jnp.sum((y_true - y_pred) ** 2)

print(loss_example(values, rngs[1]))
```

```python
# Gradient of the objective w.r.t input values

gradient = jax.grad(loss_example)(values, rngs[1])
print(gradient)
```

As above, we showcase this example on gradient descent to minimize this loss.

```python
steps = 20
values_t = values
eta = 0.1

grad_func = jax.jit(jax.grad(loss_example))

for t in range(steps):
  rngs = jax.random.split(rngs[1], 2)
  values_t = values_t - eta * grad_func(values_t, rngs[1])
```

```python
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

# General input / outputs (Pytrees)


This method can be applied to any function taking pytrees as input and output in the forward mode, and can also be used to compute derivatives, as illustrated below

```python
tree_a = (jnp.array((0.1, 0.4, 0.5)),
          {'k1': jnp.array((0.1, 0.2)),
           'k2': jnp.array((0.1, 0.1))},
          jnp.array((0.4, 0.3, 0.2, 0.1)))
```

## Tree argmax


This piecewise constant function applies the argmax to every leaf array of the pytree

```python
argmax_tree = lambda x: jax.tree.map(argmax_one_hot, x)
```

```python
argmax_tree(tree_a)
```

The perturbed approximation applies a perturbed softmax

```python
N_SAMPLES = 100
sigma = 1.0

pert_argmax_fun = perturbations.make_perturbed_fun(argmax_tree,
                                                   num_samples=N_SAMPLES,
                                                   sigma=SIGMA)
```

```python
pert_argmax_fun(rng, tree_a)
```

## Scalar loss

```python
def pert_loss(inputs, rng):
  pert_softmax = pert_argmax_fun(rng, inputs)
  argmax = argmax_tree(inputs)
  diffs = jax.tree.map(lambda x, y: jnp.sum((x - y) ** 2 / 4), argmax, pert_softmax)
  return optax.tree.sum(diffs)
```

```python
init_loss = pert_loss(tree_a, rng)

print(f'initial loss value = {init_loss:.3f}')
```

## Gradient computation


The gradient of the scalar loss can be evaluated

```python
grad = jax.grad(pert_loss)(tree_a, rng)

print('Gradient of the scalar loss')
print()
grad
```

A small step in the gradient direction reduces the value

```python
eta = 1e-1

loss_step = pert_loss(optax.tree.add_scale(tree_a, -eta, grad), rng)

print(f'initial loss value = {init_loss:.3f}')
print(f'loss after gradient step = {loss_step:.3f}')
```
