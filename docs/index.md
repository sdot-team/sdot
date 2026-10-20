---
layout: home

hero:
  name: SDOT
  text: Semi-Discrete Optimal Transport
  tagline: Fast · Differentiable · N-dimensional · Framework independent (NumPy/JAX/PyTorch/CuPy)
  image:
    src: /examples/img/patate.png
    alt: VitePress
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Examples
      link: /examples/
    - theme: alt
      text: GitHub
      link: https://github.com/sdot-team/sdot

features:
  - icon: 📦
    title: Many target distributions
    details: Images, meshes, gaussian mixtures, polytopes... with smoothing (continuation) for a robust convergence.
  - icon: 🔗
    title: Differentiable end-to-end
    details: Use sdot as a differentiable layer inside JAX jit / grad or a PyTorch autograd graph.
  - icon: 📐
    title: Any number of dimensions
    details: 2D, 6D... — the same API and the same convergence behavior.
  - icon: 🚀
    title: Highly optimized
    details: SIMD on the CPU, CUDA on the GPU, batched plans for data-parallel workloads.
  - icon: ⚡
    title: O(n log n) — not O(n²)
    details: A damped Newton method on power diagrams. Memory efficient.
---

## Hello, World

The Wasserstein distance between 100 diracs and an image:

::: code-group

<<< @/snippets/hello_numpy.py [NumPy]

<<< @/snippets/hello_jax.py [JAX]

<<< @/snippets/hello_torch.py [PyTorch]

<<< @/snippets/hello_cupy.py [CuPy]

:::

<!-- `sdot.w2` gives the distance, differentiable with respect to the diracs and the image. `sdot.plan` gives the whole plan
( barycenters, cell masses... ), and `sdot.ot_solve` the same with the settings of the solver. See
[Plans and settings](/guide/ot-plan). -->

## How it works

Given a discrete measure **f** ( a weighted sum of Dirac masses ) and a continuous density **g**, semi-discrete OT finds
the _power diagram_ ( Laguerre tessellation ) whose cells have the masses of the diracs:

$$f = \sum_i m_i \, \delta_{y_i}, \qquad \text{find } w \text{ such that } \int_{C_i(w)} g(x) \, dx = m_i \quad \forall i$$

The solution is unique, and a damped Newton method finds it in a handful of iterations, each one a power diagram plus a
linear solve, both almost linear in the number of diracs. On an 8-core CPU, 10⁵ diracs in 2D take about **1 s**; on a GPU,
**0.07 s** ( [benchmarks](/guide/performance) ).

> **Reference:** Kitagawa, Mérigot, Thibert —
> _Convergence of a Newton algorithm for semi-discrete optimal transport_, JEMS 2019.

For the ground metric, SDOT focuses essentially for now on variants of the euclidean norm.

## What's next

- [Getting started →](/guide/getting-started) installation, ...
- [Examples gallery →](/examples/)
- [Background →](/guide/background) the mathematics
- [Plans and settings →](/guide/ot-plan) barycenters, moments, ...
- [Distributions →](/guide/distributions/): images, sum of gaussians, meshes, polytopes...
- [Roadmap →](/guide/roadmap) what we are working on
