---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
---

<!-- #region id="-2Bmeh2sl_fp" -->
# Summary

The purpose of this notebook is to demonstrate example usages of utilities defined by the optax microbatching API.  microbatching is a general purpose function transformation that lifts a function that operates over a batch to one that operates over a potentially much larger batch, by splitting up the work into smaller chunks and accumulating the results.  Like other jax transformations, it's designed to be quite general - any function that can normally be traced by other jax transformations should work here. This notebook is broken up into multiple sections to illustrate usages of different functions in the API.
<!-- #endregion -->

```python id="5LUEGvWml-mH"
# ! pip install -q "optax @ git+https://github.com/google-deepmind/optax"
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import functools
import time
from optax import microbatching
import gc
```

<!-- #region id="T5Ie-_tfqgAV" -->
# Setup

Here we define a minimal transformer architecture, along with some dummy data to call it with. This notebook is primarily concerned with demonsrating the APIs
and looking at throughput numbers (examples processed / second), rather than training a model on real data. This notebook is intended to run on a Google Colab Instance with T4 GPU type.  If you run on different hardware, you may need to scale down the model size via the configuration below.
<!-- #endregion -->

```python id="nInGNjjeqcvU"
class TransformerBlock(nnx.Module):
  def __init__(self, hidden_size: int, num_heads: int, *, rngs: nnx.Rngs):
    self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
    self.mha = nnx.MultiHeadAttention(num_heads, hidden_size, rngs=rngs)
    self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)
    self.mlp = nnx.Sequential(
        nnx.Linear(hidden_size, 4*hidden_size, rngs=rngs),
        jax.nn.gelu,
        nnx.Linear(4*hidden_size, hidden_size, rngs=rngs),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    attention_output = self.mha(self.norm1(x), self.norm1(x), self.norm1(x), decode=False)
    x = x + attention_output
    mlp_output = self.mlp(self.norm2(x))
    return x + mlp_output

class Transformer(nnx.Module, pytree=False):
  def __init__(self, vocab_size: int, num_layers: int, hidden_size: int, num_heads: int = 8, *, rngs: nnx.Rngs):
    self.embedding = nnx.Embed(vocab_size, hidden_size, rngs=rngs)
    self.layers = [TransformerBlock(hidden_size, num_heads, rngs=rngs) for _ in range(num_layers)]
    self.final_layer = nnx.Linear(hidden_size, vocab_size, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = self.embedding(x)

    for layer in self.layers:
      x = layer(x)

    logits = self.final_layer(x)
    return logits
```

```python id="QEcf3p7fqoJ0"
hidden_size = 512
num_heads = 8
num_layers = 12
vocab_size = 10000
data_size = 8192
sequence_length = 256
batch_size = 32
accumulation_steps = 16

model = Transformer(vocab_size, num_layers, hidden_size, num_heads, rngs=nnx.Rngs(0))
key = jax.random.key(0)
batch = jax.random.randint(key, (batch_size, sequence_length), 0, vocab_size)

graphdef, params = nnx.split(model, nnx.Variable)
print(optax.tree.size(params))

adamw = optax.adamw(0.01)
opt_state = adamw.init(params)
```

```python id="iR9qzYp6qjDS"
def loss_fn(params, batch):
  model = nnx.merge(graphdef, params)
  logits = model(batch)
  return optax.softmax_cross_entropy_with_integer_labels(
      logits[:, :-1], labels=batch[:,1:]
  ).mean()


@functools.partial(jax.jit, donate_argnums=(0,1))
def update_fn(params, opt_state, batch):
  loss, grads = jax.value_and_grad(loss_fn)(params, batch)
  updates, opt_state = adamw.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

update_fn.lower(params, opt_state, batch).compile()
```

<!-- #region id="KlRFCRc5nHLP" -->
# Part 1: Microbatching for Gradient Accumulation

This section compares three methdos for performing gradient accumulation in jax/optax, one of which is through the microbatching API.
<!-- #endregion -->

<!-- #region id="COxGVTPUq9bZ" -->
### Option 1: Manual Gradient Accumulation

64 is the largest batch size we can use with this combination of model, sequence length, and hardware.  To use larger batch sizes we have multiple options.  The first we will explore is manual, bypassing any optax abstractions.  

Specifically, we will write two functions:
1) One that computes gradients and adds them to an accumulator.
2) One that takes accumulated gradients, performs the optimizer step, and resets the accumulated gradients back to zero.
<!-- #endregion -->

```python id="SEf4ohj-q56F"
@functools.partial(jax.jit, donate_argnums=(2,))
def add_gradient(params, batch, accumulated_gradients):
  grad = jax.grad(loss_fn)(params, batch)
  return jax.tree.map(jnp.add, grad, accumulated_gradients)


@functools.partial(jax.jit, donate_argnums=(0, 1, 2))
def update_params(params, opt_state, accumulated_gradients):
  updates, opt_state = adamw.update(accumulated_gradients, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, optax.tree.zeros_like(accumulated_gradients)

accumulated_gradients = optax.tree.zeros_like(params)
add_gradient.lower(params, batch, accumulated_gradients).compile()
update_params.lower(params, opt_state, accumulated_gradients).compile()
```

```python id="FjIvz_kRrAXH"
start_time = time.perf_counter()
for i in range(accumulation_steps):
  accumulated_gradients = add_gradient(params, batch, accumulated_gradients)

params, opt_state, accumulated_gradients = jax.block_until_ready(
    update_params(params, opt_state, accumulated_gradients)
)
end_time = time.perf_counter()
print('Total Time', end_time - start_time)
```

<!-- #region id="gPfNKr5RrHmq" -->
### Option 2: optax.MultiSteps

By wrapping our optimizer with optax.MultiSteps, we can have optax handle the gradient accumulation for us.  Now we only have to define and compile a single update_fn, which is slightly simpler. The opt_state now keeps track of the accumulated gradients for us. This is more convenient as we have only a single jitted function now.
<!-- #endregion -->

```python id="vTMm84VyrDZV"
multi_adam = optax.MultiSteps(adamw, accumulation_steps)

@functools.partial(jax.jit, donate_argnums=(0, 2))
def update_fn_v2(params, batch, opt_state):
  grads = jax.grad(loss_fn)(params, batch)
  updates, opt_state = multi_adam.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

multi_opt_state = multi_adam.init(params)
update_fn_v2.lower(params, batch, multi_opt_state).compile()
```

```python id="qi-zg9-5rGsg"
start_time = time.perf_counter()
for i in range(accumulation_steps):
  params, multi_opt_state = update_fn_v2(params, batch, multi_opt_state)

jax.block_until_ready((params, multi_opt_state))
end_time = time.perf_counter()
print('Total Time', end_time - start_time)
```

<!-- #region id="SBGm3wY5rUIH" -->
### Option 3: `microbatching.microbatch`

microbatching differs from the approach above in that it transfers the entire batch of data to device memory, then splits it up perfoming the forward-backward pass on smaller batches and accumulating them using jax.lax.scan. Like Option 2 above, the full train step can be written as a single jitted function, however now the train step is doing 16X as much work.
<!-- #endregion -->

<!-- #region id="I5HrKPuVrQiA" -->

<!-- #endregion -->

```python id="aBatfdC7rO--"
@functools.partial(jax.jit, donate_argnums=(0, 2))
def update_fn_v3(params, batch, opt_state):
  grads = microbatching.microbatch(
      jax.grad(loss_fn),
      argnums=1,
      microbatch_size=batch_size,
  )(params, batch)
  updates, opt_state = adamw.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state

full_batch = jnp.vstack([batch]*accumulation_steps)
update_fn_v3.lower(params, full_batch, opt_state).compile()
```

```python id="JKxDiYD9rXMF"
start_time = time.perf_counter()
params, opt_state = jax.block_until_ready(update_fn_v3(params, full_batch, opt_state))
end_time = time.perf_counter()
print('Total Time', end_time - start_time)
```

<!-- #region id="MZ2Xwu9znYYl" -->
# Part 2: `microbatching.micro_vmap`

micro_vmap combines microbatching with jax.vmap, providing a new transformation with a similar API as jax.vmap, but that works with much larger batches than jax.vmap. It is especially useful when the function being vmapped requries more memory than that of the inputs/outputs for intermediates, or if you want to aggregate across the vmapped dimension.
<!-- #endregion -->

```python id="fd4XmvYjnoh3"
def expensive_function(x):
  return jax.nn.softmax(jnp.sin(jnp.outer(x, x))).sum(axis=0)

B = 1024
N = 4096
X = jax.random.normal(jax.random.key(0), (B, N))

# processing more examples at a time can cause ResourceExhausted errors.
result = jax.jit(jax.vmap(expensive_function))(X[:32])
result = jax.block_until_ready(result)
gc.collect()
print('Processed small batch', result.shape)
print(result)
```

```python id="nFGg2x5erilE"
result = microbatching.micro_vmap(expensive_function, microbatch_size=32)(X)
result = jax.block_until_ready(result)
gc.collect()
print('Processed Full Batch', result.shape)
print(result)
```

<!-- #region id="PWuT3BU1xJ2S" -->
# Part 3: `microbatching.micro_grad`

micro_grad provides a simple and performant way to compute a sum or average of transformed per-example grads. While normally computing per-example gradients with jax is more expensive than computing normal gradients, and fail to run for the same batch sizes, the microbatching provides a sound mechanism to bypass this issue that we surface through the convenient and familiar API. Below we use the API to collect metrics about the per-example gradients, which can be useful for understanding and debugging the behavior of training runs.
<!-- #endregion -->

```python id="3hPNgICJuskf"
def metrics_fn(per_example_grad):
  leaf_norms = jax.tree.map(jnp.linalg.norm, per_example_grad)
  return leaf_norms

grad_fn = microbatching.micro_grad(loss_fn, metrics_fn=metrics_fn, microbatch_size=8)

grad, aux = jax.jit(grad_fn)(params, batch)
```

```python id="ph9T_WAWykls"
# This shows the norm for the gradient of the embedding layer per example.
# High uniformity of the norm values is encouraging.
aux.metrics['embedding']['embedding'].get_value()
```

<!-- #region id="XaMThtUpnakg" -->

<!-- #endregion -->
