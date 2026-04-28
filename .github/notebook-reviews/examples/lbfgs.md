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

<!-- #region id="Ch-DCRdcNnLY" -->
# L-BFGS

L-BFGS is a classical optimization method that uses past gradients and parameters information to iteratively refine a solution to a minimization problem. In this notebook, we illustrate
1. how to use L-BFGS as a simple gradient transformation,
2. how to wrap L-BFGS in a solver, and how linesearches are incorporated,
3. how to debug the solver if needed.

<!-- #endregion -->

```python id="0kUKnsM4nb5m"
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrd

import optax
import optax.tree
```

<!-- #region id="Cja5fd5FTlGs" -->
## L-BFGS as a gradient transformation

<!-- #endregion -->

<!-- #region id="UbkaJmath6Np" -->
### What is L-BFGS?

To solve a problem of the form

$$
\min_w f(w),
$$

L-BFGS ([Limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) makes steps of the form

$$
w_{k+1} = w_k - \eta_k P_k g_k,
$$

where, at iteration $k$, $w_k$ are the parameters, $g_k = \nabla f_k$ are the gradients, $\eta_k$ is the stepsize, and $P_k$ is a *preconditioning* matrix, that is, a matrix that transforms the gradients to ease the optimization process.

L-BFGS builds the preconditioning matrix $P_k$ as an approximation of the Hessian inverse $P_k \approx \nabla^2 f(w_k)^{-1}$ using past gradient and parameters information. Briefly, at iteration $k$, the previous preconditioning matrix $P_{k-1}$ is updated such that $P_k$ satisfies the secant condition $P_k(w_k-w_{k-1}) = g_k -g_{k-1}$. The original BFGS algorithm updates $P_k$ using all past information, the limited-memory variant only uses a fixed number of past parameters and gradients to build $P_k$. See [Nocedal and Wright, Numerical Optimization, 1999](https://www.math.kent.edu/~reichel/courses/optimization/Numerical_Optimization.pdf) or the [documentation](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_lbfgs) for more details on the implementation.

<!-- #endregion -->

<!-- #region id="2eoJ_CrpmgqK" -->
### Using L-BFGS as a gradient transformation

The function {py:func}`optax.scale_by_lbfgs` implements the update of the preconditioning matrix given a running optimizer state $s_k$. Given $(g_k, s_k, w_k)$, this function returns $(P_kg_k, s_{k+1})$. We illustrate its performance below on a simple convex quadratic.
<!-- #endregion -->

```python id="gySpnJ-ch5YT"
# Define objective
dim = 8
w_opt = jnp.ones(dim)
mat = jrd.normal(jrd.PRNGKey(0), (dim, dim))
mat = mat.dot(mat.T)


def fun(w):
  return 0.5 * (w - w_opt).dot(mat.dot(w - w_opt))


# Define optimizer
lr = 1e-1
opt = optax.scale_by_lbfgs()

# Initialize optimization
w = jrd.normal(jrd.PRNGKey(1), (dim,))
state = opt.init(w_opt)

# Run optimization
for i in range(16):
  v, g = jax.value_and_grad(fun)(w)
  print(f'Iteration: {i}, Value:{v:.2e}')
  u, state = opt.update(g, state, w)
  w = w - lr * u

print(f'Final value: {fun(w):.2e}')
```

<!-- #region id="f-1P4_79rnow" -->
## L-BFGS as a solver

L-BFGS is a sample in numerical optimization to solve medium scale problems. It is often the backend of generic minimization functions in software libraries like [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize). A key ingredient to make it a simple optimization blackbox, is to remove the need of tuning the stepsize, a.k.a. learning rate in machine learning. In a deterministic setting (no additional varying inputs like inputs/labels), such automatic tuning of the stepsize is done by means of linesearches reviewed below.
<!-- #endregion -->

<!-- #region id="6KIZE_pFwdcz" -->
### What are linesearches?

Given current parameters $w_k$, an update direction $u_k$ (such as the negative preconditioned gradient $u_k = -P_k g_k$ returned by L-BFGS), a linesearch computes a stepsize $\eta_k$ such that the next iterate $w_{k+1} = w_k + \eta_k u_k$ satisfies some criterions.

#### Sufficient decrease (Armijo-Goldstein criterion)

The first criterion that a good stepsize may need to satisfy is to ensure that the next iterate decreases the value of the objective by a a sufficient amount. Mathematically, the criterion is expressed as finding $\eta_k$ such that

$$
f(w_k + \eta_k u_k) \leq f(w_k) + c_1 \eta_k \langle u_k, g_k\rangle
$$

where $c_1$ is some constant set to $10^{-4}$ by default. Consider for example the update direction to be $u_k = -g_k$, i.e., moving along the negative gradient direction. In that case the criterion above reduces to $f(w_k - \eta_k g_k) \leq f(w_k) - c_1 \eta_k ||g_k||_2^2$. The criterion amounts then to choosing the stepsize such that it decreases the objective by an amount proportional to the squared gradient norm.

As long as the update direction is a *descent direction*, that is, $\langle u_k, g_k\rangle < 0$ the above criterion is guaranteed to be satisfied by some sufficiently small stepsize.
A simple linesearch technique to ensure a sufficient decrease is then to decrease a candidate stepsize by a constant factor up until the criterion is satisfied. This amounts to the backtracking linesearch implemented in {py:func}`optax.scale_by_backtracking_linesearch` and briefly reviewed below.

#### Small curvature (Strong wolfe criterion)

The sufficient decrease criterion ensures that the algorithm does not produce a sequence of diverging objective values. However, we may want to not only reduce a current stepsize but also increase it to ensure maximal speed. Ideally, we would like to find the stepsize that minimizes the function along the current update, i.e., $\eta_k^* = \arg\min_\eta f(w_k + \eta u_k)$. Such an endeavor can be computationally prohibitive, so we may select a stepsize that ensures some properties that an optimal stepsize would satisfy. In particular, we may search for a stepsize such that the derivative of $h(\eta) = f(w_k + \eta u_k)$ is small enough compared to its derivativeœ at $\eta=0$. Formally, we may want to select the stepsize $\eta_k$ such that $|h'(eta_k)| \leq |h'(0)|$, that is,

$$
|\langle \nabla f(w_k + \eta_k u_k), u_k\rangle|
\leq |\langle \nabla f(w_k), u_k\rangle|.
$$

See Chapter 3 of [Nocedal and Wright, Numerical Optimization, 1999](https://www.math.kent.edu/~reichel/courses/optimization/Numerical_Optimization.pdf) for some illustrations of this criterion. A linesearch method that can ensure both criterions require some form of bisection method implemented in optax with the {py:func}`optax.scale_by_zoom_linesearch` method. Several other linesearch techniques exist, see e.g. https://github.com/JuliaNLSolvers/LineSearches.jl. It is generally recommended to combine L-BFGS with a line-search ensuring both sufficient decrease and small curvature, which the {py:func}`optax.scale_by_zoom_linesearch` ensures.


<!-- #endregion -->

<!-- #region id="PmaDMT8O2dub" -->
### Linesearches in practice

To find a stepsize satisfying the above criterions, a linesearch needs to access the value and potentially the gradient of the function. So linesearches in optax are implemented as {py:func}`optax.GradientTransformationExtraArgs`, which take the current value, gradient of the objective as well as the function itself. We illustrate this below with {py:func}`optax.scale_by_backtracking_linesearch`.
<!-- #endregion -->

```python id="Fz3RcdDA3714"
# Objective
def fun(w):
  return jnp.sum(jnp.abs(w))


# Linesearch, comment/uncomment the desired one
linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=15)
# linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)

# Optimizer
opt = optax.chain(
    optax.sgd(learning_rate=1.0),
    # Compare with or without linesearch by commenting this line
    linesearch,
)

# Initialize
w = jrd.normal(jrd.PRNGKey(0), (8,))
state = opt.init(w)

# Run optimization
for i in range(16):
  v, g = jax.value_and_grad(fun)(w)
  print(f'Iteration: {i}, Value:{v:.2e}')
  u, state = opt.update(g, state, w, value=v, grad=g, value_fn=fun)
  w = w + u

print(f'Final value: {fun(w):.2e}')
```

<!-- #region id="CLShtM0s3TAP" -->
To validate the stepsize the linesearch calls the function several times. If a stepsize is accepted, we have then a priori access to the value of the function, and, potentially its gradient. The implementation of the linesearches in optax store the value and the gradient computed by the linesearch to avoid recomputing them at the next step. In practice, the code above can be modified as follows.

*Note:*
The backtracking linesearch only evaluates the function and does not compute the gradient natively. To make the backtracking linesearch compute and store the gradient at the stepsize taken, we add the flag `store_grad=True`, see below.
The zoom linesearch always compute both function and gradient so there is no need to specify an additional flag.
<!-- #endregion -->

```python id="RFH5Llz06iwX"
# Objective
def fun(w):
  return jnp.sum(jnp.abs(w))


# Linesearch
linesearch = optax.scale_by_backtracking_linesearch(
    max_backtracking_steps=15, store_grad=True
)
# linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)

# Optimizer
opt = optax.chain(optax.sgd(learning_rate=1.0), linesearch)

# Initialize
w = jrd.normal(jrd.PRNGKey(0), (8,))
state = opt.init(w)

# Run optimization
for _ in range(16):
  # Replace `v, g = jax.value_and_grad(fun)(w)` by
  v, g = optax.value_and_grad_from_state(fun)(w, state=state)
  u, state = opt.update(g, state, w, value=v, grad=g, value_fn=fun)
  w = w + u

print(f'Final value: {fun(w):.2e}')
```

<!-- #region id="Fqylj3ANOiYa" -->
### L-BFGS solver

Optax combines then the gradient transformation of L-BFGS and a linesearch in `optax.lbfgs()`.

We present below a wrapper that combines both into a solver which tries to find the minimizer of a function given
1. some initial parameters `init_params`,
2. the function to optimize `fun`,
3. the instance of the L-BFGS solver considered `opt`,
4. a maximal number of iteration `max_iter`,
5. a tolerance `tol` on the optimization error measured here as the gradient norm.
<!-- #endregion -->

```python id="3BM2rlAGUx7K"
def run_opt(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = optax.tree.get(state, 'count')
    grad = optax.tree.get(state, 'grad')
    err = optax.tree.norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state
```

<!-- #region id="Ohk5KQqfZgWt" -->
We can test the solver on the [Rosenbrock function](https://www.sfu.ca/~ssurjano/rosen.html).
<!-- #endregion -->

```python id="bVrZLiSVZfue"
def fun(w):
  return jnp.sum(100.0 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2)

opt = optax.lbfgs()
init_params = jnp.zeros((8,))
print(
    f'Initial value: {fun(init_params):.2e} '
    f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(init_params)):.2e}'
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=100, tol=1e-3)
print(
    f'Final value: {fun(final_params):.2e}, '
    f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
)
```

<!-- #region id="KinZFIXxbBxy" -->
We may add additional information by simply chaining `optax.lbfgs` with an identity transform that just prints relevant information as follows.
<!-- #endregion -->

```python id="YB4NqbThbdo7"
class InfoState(NamedTuple):
  iter_num: jax.typing.ArrayLike


def print_info():
  def init_fn(params):
    del params
    return InfoState(iter_num=0)

  def update_fn(updates, state, params, *, value, grad, **extra_args):
    del params, extra_args

    jax.debug.print(
        'Iteration: {i}, Value: {v}, Gradient norm: {e}',
        i=state.iter_num,
        v=value,
        e=optax.tree.norm(grad),
    )
    return updates, InfoState(iter_num=state.iter_num + 1)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def fun(w):
  return jnp.sum(100.0 * (w[1:] - w[:-1] ** 2) ** 2 + (1.0 - w[:-1]) ** 2)


opt = optax.chain(print_info(), optax.lbfgs())
init_params = jnp.zeros((8,))
print(
    f'Initial value: {fun(init_params):.2e} '
    f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(init_params)):.2e}'
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=100, tol=1e-3)
print(
    f'Final value: {fun(final_params):.2e}, '
    f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
)
```

<!-- #region id="KZIu7UDveO6D" -->
## Debugging


<!-- #endregion -->

<!-- #region id="LV8CslWpoDDq" -->
### Accessing debug information

In some cases, L-BFGS with a linesearch as a solver will fail. Most of the times, the culprit goes down to the linesearch. To debug the solver in such cases, we provide a `verbose` option to the `optax.scale_by_zoom_linesearch`. We show below how to proceed.

To demonstrate such bug, we try to minimize the [Zakharov function](https://www.sfu.ca/~ssurjano/zakharov.html) and set the `scale_init_precond` option to `False` (by choosing the default option `scale_init_precond=True`, the algorithm would actually run fine, we just want to showcase the possibility to use debugging in the linesearch here). You'll observe that the final value is the same as the initial value which points out that the solver failed.
<!-- #endregion -->

```python id="NbyxeORif9wf"
def fun(w):
  ii = jnp.arange(1, len(w) + 1, step=1, dtype=w.dtype)
  sum1 = (w**2).sum()
  sum2 = (0.5 * ii * w).sum()
  return sum1 + sum2**2 + sum2**4

opt = optax.lbfgs(scale_init_precond=False)

init_params = jnp.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e4])
print(
    f'Initial value: {fun(init_params)} '
    f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(init_params))}'
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=50, tol=1e-3)
print(
    f'Final value: {fun(final_params)}, '
    f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params))}'
)
```

<!-- #region id="uwcbY5UXohZB" -->
The default implementation of the linesearch in the code is
```
scale_by_zoom_linesearch(max_linesearch_steps=20, initial_guess_strategy='one')
```
To debug we can set the verbose option of the linesearch to `True`.
<!-- #endregion -->

```python id="j8FUTkc8o3l2"
opt = optax.chain(print_info(), optax.lbfgs(scale_init_precond=False,
  linesearch=optax.scale_by_zoom_linesearch(
      max_linesearch_steps=20, verbose=True, initial_guess_strategy='one'
  )
))

init_params = jnp.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e4])
print(
    f'Initial value: {fun(init_params):.2e} '
    f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(init_params)):.2e}'
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=100, tol=1e-3)
print(
    f'Final value: {fun(final_params):.2e}, '
    f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
)
```

<!-- #region id="nCgpjzCbo7p9" -->
As expected, the linesearch failed at the very first step taking a stepsize that did not ensure a sufficient decrease. Multiple information is displayed. For example, the slope (derivative along the update direction) at the first step is extremely large which explains the difficulties to find an appropriate stepsize. As pointed out in the log above, the first thing to try is to use a larger number of linesearch steps.
<!-- #endregion -->

```python id="nA9WVXykpaKf"
opt = optax.chain(print_info(), optax.lbfgs(scale_init_precond=False,
  linesearch=optax.scale_by_zoom_linesearch(
      max_linesearch_steps=50, verbose=True, initial_guess_strategy='one'
  )
))

init_params = jnp.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e4])
print(
    f'Initial value: {fun(init_params):.2e} '
    f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(init_params)):.2e}'
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=100, tol=1e-3)
print(
    f'Final value: {fun(final_params):.2e}, '
    f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
)
```

<!-- #region id="na-7s1Q2o1Rc" -->
By simply taking a maximum of 50 steps of the linesearch instead of 20, we ensured that the first stepsize taken provided a sufficient decrease and the solver worked well.
Additional debugging information can be found in the source code accessible from the docs of {py:func}`optax.scale_by_zoom_linesearch`.
<!-- #endregion -->

<!-- #region id="74ZbgzcKoJ0J" -->
### Tips

- **LBFGS**
  - Selecting a higher `memory_size` in lbfgs may improve performance at a memory and computational cost. No real gains may be perceived after some value.
  - `scale_init_precond=True` is standard. It captures a similar scale as other well-known optimization methods like Barzilai Borwein.

- **Zoom linesearch**
  - Remember there are two conditions to be met (sufficient decrease and small curvature). If the algorithm takes too many linesearch steps, you may try
  setting `curv_rtol = jnp.inf`, effectively ignoring the small curvature condition. The resulting algorithm will essentially perform a backtracking linesearch where a valid stepsize is searched by minmizing a quadratic or cubic approximation of the objective (so that would be a potentially faster algorithm than the current implementation of `scale_by_backtracking_linesearch`).
  - As pointed above, if the solver gets stuck, try using a larger number of linesearch steps and print debugging information.

You may run the solver in double precision by setting `jax.config.update("jax_enable_x64", True)`. If you use double precision, consider augmenting the number of linesearch steps to reach the machine precision (like using `max_linesearch_steps=55`).

<!-- #endregion -->

<!-- #region id="T-oGa3P2sCbH" -->
## Contributing and benchmarking

Numerous other linesearch could be implemented, as well as other solvers for medium scale problems without stochasticity. Contributions are welcome.

If you want to contribute a new solver for medium scale problems like LBFGS, benchmarks would be highly appreciated. We provide below an example of benchmark (which could also be used if you want to test some hyperparameters of the algorithm). We take here the classical Rosenbroke function, but it could be better to expand such benchmarks to e.g. the set of test functions given by [Andrei, 2008](https://camo.ici.ro/journal/vol10/v10a10.pdf).
<!-- #endregion -->

```python id="MagDCuGjsB5x"
import time
num_fun_calls = 0

def register_call():
  global num_fun_calls
  num_fun_calls += 1

def test_hparams(lbfgs_hparams, linesearch_hparams, dimension=512):
  global num_fun_calls
  num_fun_calls = 0

  def fun(x):
    jax.debug.callback(register_call)
    return jnp.sum((x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

  opt = optax.chain(optax.lbfgs(**lbfgs_hparams,
    linesearch=optax.scale_by_zoom_linesearch(**linesearch_hparams)
    )
  )

  init_params = jnp.arange(dimension, dtype=jnp.float32)

  tic = time.time()
  final_params, _ = run_opt(
      init_params, fun, opt, max_iter=500, tol=5*1e-5
    )
  final_params = jax.block_until_ready(final_params)
  time_run = time.time() - tic

  final_value = fun(final_params)
  final_grad_norm = optax.tree.norm(jax.grad(fun)(final_params))
  return final_value, final_grad_norm, num_fun_calls, time_run

```

```python id="7CXMxWsztGf5"
import copy
import matplotlib.pyplot as plt

default_lbfgs_hparams = {'memory_size': 15, 'scale_init_precond': True}
default_linesearch_hparams = {
    'max_linesearch_steps': 15,
    'initial_guess_strategy': 'one'
}

memory_sizes = [int(2**i) for i in range(7)]
times = []
calls = []
values = []
grad_norms = []
for m in memory_sizes:
  lbfgs_hparams = copy.deepcopy(default_lbfgs_hparams)
  lbfgs_hparams['memory_size'] = m
  v, g, n, t = test_hparams(lbfgs_hparams, default_linesearch_hparams, dimension=1024)
  values.append(v)
  grad_norms.append(g)
  calls.append(n)
  times.append(t)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].plot(memory_sizes, values)
axs[0].set_ylabel('Final values')
axs[0].set_yscale('log')
axs[1].plot(memory_sizes, grad_norms)
axs[1].set_ylabel('Final gradient norms')
axs[1].set_yscale('log')
axs[2].plot(memory_sizes, calls)
axs[2].set_ylabel('Number of function calls')
axs[3].plot(memory_sizes, times)
axs[3].set_ylabel('Run times')
for i in range(4):
  axs[i].set_xlabel('Memory size')
plt.tight_layout()
```
