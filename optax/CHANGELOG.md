# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Removed

- Stochastic gradient estimators à la Reinforce with control variates methods.
See monte_carlo folder in optax 0.1.8 if you are interested.
- Removed optax._src.transform.cast_tree and optax._src.utils.cast_tree. Use
optax.tree.cast from now on.
