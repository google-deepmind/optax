# 🛠️ Development

Optax welcomes contributions from the open-source community. This can include
issues, bug reports, questions, design documents, pull requests, or any other
input to the project.

## How to Contribute

For most contributions, the best way to get started is to raise an issue in the
Github [issue tracker](https://github.com/deepmind/optax/issues) describing the
problem to be solved, or the idea to be worked on. This will enable some
discussion on the best way to land new features, and can also provide
opportunities for collaborations with other contributors.

Some more details on contributing code are provided in the
[CONTRIBUTING.md](https://github.com/deepmind/optax/blob/main/CONTRIBUTING.md)
file in the source tree.


(inclusion_criteria)=
## Inclusion Criteria

We only consider well-established algorithms for inclusion in the main `optax`
package. A rule of thumb is at least 2 years since publication, 100+ citations,
and wide usefulness. A small modification of an existing algorithm that provides
a clear-cut improvement on a widely-used method will also be considered for
inclusion.

Algorithms that don't meet these criteria should instead be submitted to the
{doc}`api/contrib` directory. When in doubt, we recommend submitting new
algorithms to this directory.


#### Design Documents

For more complex or involved features, we recommend starting by writing a
design document, or RFC ("Request For Comments") before spending significant
time writing code. This can provide an opportunity for other contributors to
provide input and find the best way to land new features and also provides a
reference for future users to understand how the library works.

For an example of this, see the following
[[RFC] Proposal for complex-valued optimization in Optax](https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29) authored by [Dian [Wu](https://github.com/wdphy16) which led to the addition of improved [complex numbers](https://optax.readthedocs.io/en/latest/api/contrib.html?complex-valued-optimization#complex-valued-optimization) support in Optax.


#### Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

#### Ideas for Contributions

If you would like to get started with contributing to Optax, but do not know
what to start working on, a selection of *"good first issues"* for new
contributors are given in the
[issue tracker](https://github.com/deepmind/optax/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue).
Ideas for good starter contributions are also welcomed.

#### Improving the documentation

If you would like to help contributing to the documentation, install the
required packages by running `pip install .[docs]`.
Then, to build the docs, from the docs folder, run `make html` to build all docs
and notebooks or `make html-noplot` to build the docs without executing
the notebooks (much faster).

## Core Maintainers

*   [Iurii Kemaev](https://github.com/hbq1)
*   [Markus Kunesch](https://github.com/mkunesch)
*   [Matteo Hessel](https://github.com/mtthss)
*   [Ross Hemsley](https://github.com/rosshemsley)

## Collaborators

We'd also like to extend a special thanks to the following open source
contributors who have made significant contributions to Optax,

*   [n2cholas](https://github.com/n2cholas)
*   [wdphy16](https://github.com/wdphy16)
*   [holounic](https://github.com/holounic)

A full list of open source contributors can be found
[here](https://github.com/deepmind/optax/graphs/contributors).

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
