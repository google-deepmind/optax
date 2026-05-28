# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
