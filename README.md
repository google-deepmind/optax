# Optax

Optax is a composable gradient processing and optimization library for JAX.

## Transformations

Optax implements a number of composable gradient transformations,
typically used in the context of optimizing deep neural networks.

Each transformation defines:

* ``init_fn``: initialize (possibly empty) sets of statistics (aka ``state``),
* ``update_fn``: transform a candidate gradient based update.

Examples of gradient transformations are 1) clipping gradients 2) rescaling
gradients (e.g. adam, rmsprop) 3) adding gradient noise, etc ...

## Composing transformations

An (optional) ``chain`` utility can be used to build custom optimizers by
chaining arbitrary sequences of transformations. For any sequence of
transformations ``chain`` returns a single ``init_fn`` and ``update_fn``.

## Applying updates

An ``apply_updates`` function can be used to eventually apply the
transformed gradients to the set of parameters of interest.

Separating gradient transformations from the parameter update allows to flexibly
chain a sequence of transformations of the same gradients, as well as combine
multiple updates to the same parameters (e.g. in multi-task settings where the
different tasks may benefit from different sets of gradient transformations).

## Canonical optimisers

Many popular optimizers can be implemented using ``optax`` as one-liners, and,
for convenience, we provide aliases for some of the most popular ones.

## Schedules

Schedules allow to anneal scalar quantities over time (e.g. learning rates).

## Second Order

Computing the Hessian or Fisher information matrices for neural networks is
typically intractible due to the quadratic memory requirements. Solving for the
diagonals of these matrices is often a better solution. The library offers
functions for computing these diagonals with sub-quadratic memory requirements.
