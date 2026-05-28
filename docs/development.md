# 🛠️ Development

We welcome contributions to Optax! Whether you have a bug report, a question,
a feature suggestion, or code to share, your input is valuable.

## How to Contribute

For most contributions, the best way to get started is to raise an issue in the
Github [issue tracker](https://github.com/deepmind/optax/issues) describing the
problem to be solved, or the idea to be worked on. This will enable some
discussion on the best way to land new features, and can also provide
opportunities for collaborations with other contributors.

Some more details on contributing code are provided in the
[CONTRIBUTING.md](https://github.com/google-deepmind/optax/blob/main/CONTRIBUTING.md)
file in the source tree.

**Need ideas?** If you would like to get started with contributing to Optax,
but do not know what to start working on, check out our selection of
[*good first issues*](https://github.com/google-deepmind/optax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

**Generic template.** When contributing with a new function or optimizer, use
the following template for the docstring (see
[optax.adam](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam)
for example)
- One-line description
- Longer description with mathematic description if possible
- Args section
- Returns section
- Examples
- References
- Additional notes, warnings

The docs need to be in
[reStructuredText format](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)!
For example references should be written like
```Doe et al `Yet another optimizer <link to optimizer>`_, 2042)```
(so not in markdown format), same for
notes and warnings.

A longer description, references and additional notes may be optional. But
examples are generally not, they are the best piece of documentation.

### Can I contribute AI generated code?

All submissions to Google Open Source projects need to follow Google's
[Contributor License Agreement (CLA)](https://cla.developers.google.com/), in
which contributors agree that their contribution is an original work of
authorship. This doesn’t prohibit the use of coding assistance tools, but what’s
submitted does need to be a contributor's original creation.

In the Optax project, a main concern with AI-generated contributions is that
**low-quality AI-generated code imposes a disproportionate review cost**.
Since the team's capacity for code review is limited, we have a higher bar
for accepting AI-generated contributions compared to those written by a human.

A loose rule of thumb: if the team needs to spend more time reviewing a
contribution than the contributor spends generating it, then the contribution
is probably not helpful to the project, and we will likely reject it.

## Improving the documentation

Documentation is key, and we're particularly happy to accept documentation improvements.

Our documentation is written in [Sphinx](https://www.sphinx-doc.org/en/master/). You can
build the documentation locally as follows:

1. **Install Requirements**: `pip install -e ".[docs]"`
2. **Build the Docs**: From the `docs` folder, run:
   * `make html` (builds everything)
   * `make html-noplot` (faster, skips running examples)


### Running doctest
You can add examples illustrating how to use the functions in docstrings. For
inspiration see the `Examples:` section of the code source of `adam` in
`optax/_src/alias.py`.

To test locally such examples, run
`python -m doctest -v <path_to_your_file>.py`.

(inclusion_criteria)=
## Inclusion Criteria

To ensure Optax remains a focused and high-quality library, we have specific
guidelines for including algorithms in the main `optax` package:

1. **Established**: Algorithms should generally be published for at least 2
years, well-cited (100+ citations), and demonstrate broad utility.
2. **Significant Improvement**: Minor modifications will be considered
if they offer clear advantages over widely used methods.

If your algorithm doesn't meet the main package criteria, the {doc}`api/contrib`
directory is perfect for sharing innovative work. Please make sure that all
common tests (in `optax/contrib/_common_test.py` or `optax/_src/alias_test.py`)
are passed when you propose a new algorithm. These tests ensure the
interoperability of algorithms with different features of optax (such as
gradient accumulation or varying hyperparameters).


## Design Documents

For more complex or involved features, we recommend starting by writing a
design document, or RFC ("Request For Comments") before spending significant
time writing code. This can provide an opportunity for other contributors to
provide input and find the best way to land new features and also provides a
reference for future users to understand how the library works.

For an example of this, see the following
[[RFC] Proposal for complex-valued optimization in Optax](https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29) authored by [Dian Wu](https://github.com/wdphy16) which led to the addition of
improved
[complex numbers](https://optax.readthedocs.io/en/latest/api/contrib.html?complex-valued-optimization#complex-valued-optimization) support in Optax.


## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.


## The Team

Optax is developed by a team of researchers at Google DeepMind and Alphabet, as
well as a growing community of open-source contributors. The work on Optax is
part of a wider effort to contribute to making the
[JAX Ecosystem](https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/)
the best possible environment for ML/AI research. Below is the list of the top
(in number of commits) 30 contributors to date:


```{eval-rst}
.. contributors:: google-deepmind/optax
    :avatars:
    :limit: 30
    :order: ASC
```

A full list of the contributors to date can be found
[here](https://github.com/deepmind/optax/graphs/contributors).


## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
