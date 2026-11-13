---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: venv
    language: python
    name: python3
---

# Using the Muon Optimizer in Optax

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/contrib/muon.ipynb)

This notebook demonstrates how to use the `optax.contrib.muon` optimizer. We'll cover three main use cases:

1.  **Default Muon:** Automatically applying Muon to 2D matrices and AdamW to all other parameters.
2.  **Masked Muon:** Using `muon_weight_mask` to explicitly select which parameters are optimized by Muon.
3.  **Muon with Reshaping:** Using `muon_weight_specs` to apply Muon to higher-dimensional parameters (tensors) by specifying how they should be reshaped.

```python
from pprint import pprint

import jax
import jax.numpy as jnp
from jax import random

import optax
```

```python
# Create a sample PyTree of parameters with different dimensions
keys = iter(random.split(random.key(0), 1024))
params = {
    "layer1": {
        "w": jax.random.normal(next(keys), (128, 64)),  # 2D matrix
        "b": jax.random.normal(next(keys), (64,)),  # 1D vector
    },
    "layer2": {
        "w": jax.random.normal(next(keys), (64, 32)),  # 2D matrix
    },
    "layer3_conv": {
        "w": jax.random.normal(next(keys), (4, 3, 3, 16))  # 4D tensor
    },
}


# A simple loss function: sum of squares of parameters.
# The gradient of this loss is just the parameters themselves.
@jax.jit
def loss_fn(p):
    return sum(jnp.sum(x**2) for x in jax.tree.leaves(p))
```

```python
def print_state(state):
    print(
        "State variables using the muon transform ---------------------------"
    )
    pprint(
        {
            "".join(map(str, k)): "MUON"
            for k, v in jax.tree.flatten_with_path(state.inner_states["muon"])[
                0
            ]
            if v.ndim > 0 and not str(k[-1]).endswith("ns_coeffs")
        }
    )
    print()
    print(
        "State variables using the adam transform ---------------------------"
    )
    pprint(
        {
            "".join(map(str, k)): "ADAM"
            for k, v in jax.tree.flatten_with_path(state.inner_states["adam"])[
                0
            ]
            if v.ndim > 0 and not str(k[-1]).endswith("ns_coeffs")
        }
    )
```

## 1. Default Muon Configuration

By default, `muon` partitions parameters based on their dimensionality. Parameters with `ndim == 2` (matrices) are optimized with Muon, while all others are handled by a standard AdamW optimizer.

```python
# Use muon with default partitioning (ndim == 2 for muon)
opt = optax.contrib.muon(learning_rate=1e-3)
opt_state = opt.init(params)

print_state(opt_state)
```

## 2. Using `muon_weight_dimension_numbers` for Explicit Selection and Higher-Rank Tensors

The core Muon algorithm (specifically, the Newton-Schulz iteration) operates on 2D matrices. To apply it to tensors of rank > 2, you must provide a `MuonDimensionNumbers` that tells the optimizer how to reshape the tensor into a 2D matrix (`(reduction_dim, output_dim)`).

- `reduction_axes`: A tuple of axis indices that will be flattened into the first dimension of the matrix.
- `output_axes`: A tuple of axis indices that will be flattened into the second dimension.

Any remaining axes are treated as batch dimensions, and the operation is applied independently across them.


You can override the default behavior using `muon_weight_dimension_numbers`. This is a PyTree with the same (or a prefix) structure as your parameters, containing `MuonDimensionNumbers` named tuples. If a leaf is a `MuonDimensionNumbers` tuple, the corresponding parameter is handled by Muon; if `None`, it's handled by AdamW.

Let's apply Muon *only* to `'layer1'`'s weights and use AdamW for everything else, including the other 2D matrix in `'layer2'`.

```python
print("optax.contrib.MuonDimensionNumbers doctring:\n")
print(optax.contrib.MuonDimensionNumbers.__doc__)
```

```python
# Mask to apply Muon ONLY to layer1's weights.
weight_dim_nums = {
    "layer1": {
        # default for 2D is `optax.contrib.MuonDimensionNumbers(0, 1)`
        "w": optax.contrib.MuonDimensionNumbers(),
        "b": None,
    },
    "layer2": {
        "w": None,
    },
    "layer3_conv": {
        "w": None,
    },
}

opt = optax.contrib.muon(
    learning_rate=1e-3, muon_weight_dimension_numbers=weight_dim_nums
)
opt_state = opt.init(params)
print_state(opt_state)
```

Let's apply Muon to our 4D convolutional weight tensor from `layer3_conv`.

```python
# We want to apply Muon to the 4D convolutional kernel in 'layer3_conv'.
# The shape is (4, 3, 3, 16). Let's treat the first three axes (4*3*3=36)
# as the 'reduction' dimension and the last axis (16) as the 'output' dimension.

#  Define the corresponding MuonDimensionNumbers for the selected tensors.
#  The structure must match parameters. Use None for non-Muon params.
weight_dim_nums = {
    "layer1": {"w": optax.contrib.MuonDimensionNumbers((0,), (1,)), "b": None},
    "layer2": {"w": None},
    "layer3_conv": {
        "w": optax.contrib.MuonDimensionNumbers(
            reduction_axis=(0, 1, 2), output_axis=(3,)
        ),
    },
}

opt = optax.contrib.muon(
    learning_rate=1e-3, muon_weight_dimension_numbers=weight_dim_nums
)
opt_state = opt.init(params)

print_state(opt_state)
```
