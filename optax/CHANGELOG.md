# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.9] - 2026-11-11

### Added

- `optax.adabelief` now accepts optional `weight_decay` and `weight_decay_mask`
arguments for decoupled weight decay, following `optax.adamw`.
- `optax.contrib.prodigy` and `optax.contrib.dadapt_adamw` now accept a
`weight_decay_mask` argument (a full or prefix tree of booleans, or a callable).
- `optax.sm3` now accepts an `eps` argument.
- `optax.ArrayTree` is now part of the public API.
- `optax.microbatching` functions now support empty (size 0) batches, accept
batched arguments passed by position or by name regardless of whether they are
listed in `argnums` or `argnames`, and add `jax.named_call` annotations for
easier profiling.

### Changed

- Optimizers that add `eps` to gradient statistics (e.g. `adam`, `rmsprop`,
`adafactor`, `sm3` and several `optax.contrib` optimizers) now raise a
`ValueError` at trace time if a concrete `eps` (or `eps_root`) is positive but
smaller than the smallest normal number of the optimizer state dtype, since such
an `eps` is flushed to zero and no longer prevents division by zero (e.g.
zero-gradient `adam` steps become `NaN`). Default `eps` values are unchanged, so
e.g. `optax.adam` with the default `eps=1e-8` now raises for `float16` optimizer
state: pass a larger `eps` (e.g. `eps=1e-4`) or keep the optimizer state in
`bfloat16`/`float32`. Traced `eps` values (e.g. from `optax.inject_hyperparams`)
are not checked.
- `optax.scale_by_yogi` (and therefore `optax.yogi`) now initializes the first
moment to zero; `initial_accumulator_value` only applies to the second moment.
This slightly changes the numerical results of `optax.yogi`.
- `optax.losses.ntxent` now computes pairwise cosine similarities in
`float32`/`float64` with a 2D matrix multiplication instead of an `[N, N, D]`
broadcast, which reduces memory usage and improves numerical stability for
`bfloat16`/`float16` embeddings and small temperatures.
- Reduced the peak memory usage of `optax.losses.sigmoid_focal_loss`.
- `optax.contrib.muon` no longer unrolls its Newton-Schulz iterations, which
reduces peak memory usage without changing results.
- Examples were migrated from `flax.linen` to `flax.nnx` and no longer import
TensorFlow directly (some still load data with TensorFlow Datasets).

### Deprecated

- The `seed` argument of `optax.contrib.dpsgd` is deprecated in favor of a new
`key` argument, which accepts an integer seed or a typed PRNG key created with
`jax.random.key`.

### Removed

- Removed the legacy `optax._src.utils.safe_norm` alias. Use `optax.safe_norm`
instead.

### Fixed

- `optax.losses.cosine_similarity` and `optax.losses.cosine_distance` now apply
the `where` mask before normalizing, so `nan`/`inf` values in masked-out entries
no longer produce `NaN` gradients.
- Differentiating through `optax.clip_by_global_norm` no longer produces `NaN`
gradients when all updates are zero.
- `optax.losses.ctc_loss` and `optax.losses.ctc_loss_with_forward_probs` no
longer return slightly negative losses for confident alignments in `float32`.
- Fixed JAX version checks that compared version strings lexicographically. On
JAX 0.10 and newer they selected legacy code paths, so `optax.tree.vdot` (used
by e.g. `optax.lbfgs`, the line searches and `optax.contrib.prodigy`) and
`optax.microbatching` failed with explicit sharding.
- The error message of `optax.cosine_onecycle_schedule` now names the correct
schedule.

## [0.2.8] - 2026-03-20

### Changed

- Following the JAX 0.9.2 release, the `jax_pmap_shmap_merge` config flag was
removed so that the `jax.pmap` implementation is always based on `jax.jit` and
`jax.shard_map`, and opting into the old `jax.pmap` behavior is no longer an
option. Optax had opted into the old behavior to give users time to migrate, and
as of Optax 0.2.8 this is no longer supported. This changed shouldn't impact
most users, but if you experience errors or performance regressions as a result
of it, you can update your code following JAX's
[migration guide](https://docs.jax.dev/en/latest/migrate_pmap.html) (or use
JAX 0.9.2 or earlier and set
`jax.config.update("jax_pmap_shmap_merge", False)`).

## [0.2.7] - 2026-02-05

### Added

- CHANGELOG.md file added to track notable changes.

### Changed

- Test classes now inherit from `absl.TestCase` or `parameterized.TestCase`
instead of `chex.TestCase` as part of our effort to remove the `chex`
dependency. This means that Chex test variants (with/without `jit`, with/without
`device_put`, with `pmap`) are no longer tested. We decided it was sufficient to
use `jit` throughout the tests. There is already test coverage on both CPU and
accelerators, and `pmap` is deprecated.
- Classification losses (`poly_loss_cross_entropy`,
`ctc_loss_with_forward_probs`, `ctc_loss`, `sigmoid_focal_loss`) and regression
losses (`huber_loss`, `cosine_similarity`, `cosine_distance`) no longer support
positional args for hyperparameter-like inputs.
- Substituted `chex` types with internally-defined types and removed `chex`
dependency.

### Removed

- Stochastic gradient estimators à la Reinforce with control variates methods.
See monte_carlo folder in optax 0.1.8 if you are interested.
- Removed optax._src.transform.cast_tree and optax._src.utils.cast_tree. Use
optax.tree.cast from now on.
