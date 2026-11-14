<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

# Optax

![CI status](https://github.com/google-deepmind/optax/actions/workflows/tests.yml/badge.svg?branch=main)
[![Documentation Status](https://readthedocs.org/projects/optax/badge/?version=latest)](http://optax.readthedocs.io)
![pypi](https://img.shields.io/pypi/v/optax)

## 简介 (Introduction)

Optax 是面向 JAX 的梯度处理与优化算法库。

Optax 旨在通过提供可灵活重新组合的基础构建块（Building Blocks），来促进科学研究。

我们的目标是：

*   提供核心组件简单、经过充分测试且高效的实现。
*   通过支持将底层组件轻松组合为自定义优化器（或其他梯度处理组件），提升科研生产力。
*   让任何人都能轻松参与贡献，从而加速新思路与算法的采纳。

我们倡导专注于可有效组合为定制解决方案的微型可组合构建块。其他开发者可在这些基础组件之上构建更复杂的抽象。只要合理，在代码实现上优先考虑可读性并使其贴合标准数学公式，而非单纯追求代码复用。

该库的初始原型曾以 `jax.experimental.optix` 的形式在 JAX 的 experimental 目录中提供。鉴于 `optix` 在 DeepMind 内部得到广泛应用，并在对 API 进行了数轮迭代后，`optix` 最终移出 `experimental` 成为独立的开源库，并更名为 `optax`。

Optax 的官方文档可在 [optax.readthedocs.io](https://optax.readthedocs.io/) 查阅。

## 安装 (Installation)

你可以通过 PyPI 安装 Optax 的最新发布版本：

```sh
pip install optax
```

或者从 GitHub 安装最新的开发版本：

```sh
pip install git+https://github.com/google-deepmind/optax.git
```

## 快速上手 (Quickstart)

Optax 提供了[多种流行优化器](https://optax.readthedocs.io/en/latest/api/optimizers.html)和[损失函数](https://optax.readthedocs.io/en/latest/api/losses.html)的实现。
例如，以下代码片段使用来自 `optax.adam` 的 Adam 优化器以及来自 `optax.l2_loss` 的均方误差损失函数。我们使用 `init` 函数和模型的参数 `params` 来初始化优化器状态：

```python
optimizer = optax.adam(learning_rate)
# 获取包含优化器统计信息的 `opt_state`
params = {'w': jnp.ones((num_weights,))}
opt_state = optimizer.init(params)
```

为了编写更新循环，我们需要一个可由 JAX 进行自动微分（在本例中使用 `jax.grad`）的损失函数来计算梯度：

```python
compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
grads = jax.grad(compute_loss)(params, xs, ys)
```

随后通过 `optimizer.update` 转换梯度，得到应用于当前参数以生成新参数的更新量。`optax.apply_updates` 是执行此操作的便捷工具函数：

```python
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

你可以通过 [Optax 🚀 入门示例 Notebook](https://github.com/google-deepmind/optax/blob/main/docs/getting_started.ipynb) 继续了解快速上手详情。

## 开发 (Development)

我们欢迎提交 Issue 报告以及解决问题或改进现有功能的 Pull Request。如果您有意添加新功能（例如新的优化器），**请先提出一个 Issue**！我们致力于让 Optax 更加灵活、通用且易用，方便您定义自己的优化器。

### 源代码 (Source code)

你可以使用以下命令获取最新源代码：

```sh
git clone https://github.com/google-deepmind/optax.git
```

### 测试 (Testing)

运行测试套件，请执行以下脚本：

```sh
sh test.sh
```

### 文档构建 (Documentation)

构建文档前，首先确保安装所有相关依赖项：

```sh
pip install -e ".[docs]"
```

随后执行以下命令：

```sh
make html -C docs
```

### 基准测试 (Benchmarking)

精选基准测试：

- [Benchmarking Neural Network Training Algorithms, Dahl G. et al, 2023](https://arxiv.org/pdf/2306.07179)（神经网络训练算法基准评估），

- [Descending through a Crowded Valley — Benchmarking Deep Learning Optimizers, Schmidt R. et al, 2021](https://proceedings.mlr.press/v139/schmidt21a)（深度学习优化器基准评估）。

构建自己的基准测试：

- [Benchopt: Reproducible, efficient and collaborative optimization benchmarks, Moreau T. et al, 2022](https://arxiv.org/abs/2206.13424)（Benchopt：可复现、高效且协作的优化基准框架）。

优化器调优指南手册：

- [Deep Learning Tuning Playbook, Godbole V. et al, 2023](https://github.com/google-research/tuning_playbook)（深度学习调优指南手册）。

### JAX 中其它优化相关的库 (Other optimization-adjacent libraries in JAX)

- [optimistix](https://github.com/patrick-kidger/optimistix)：非线性求解器：求根、极小化、不动点与最小二乘法。

- [matfree](https://github.com/pnkraemer/matfree)：无矩阵算法，用于研究深度学习中的曲率动态特性。

## 引用 Optax (Citing Optax)

本仓库是 DeepMind JAX 生态系统的一部分，如需引用 Optax，请使用以下 BibTeX 格式：

```bibtex
@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/google-deepmind},
  year = {2020},
}
```

---

> 💡 **中文文档维护声明**：本中文文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）协同维护并持续跟踪上游更新。若发现翻译疏漏或有最新功能改进建议，欢迎提交 PR 或 Issue。
