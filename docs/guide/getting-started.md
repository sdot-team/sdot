# Getting Started

## Installation

```bash
pip install sdot
```

Alternatively, to get the latest version (or to participate), one can use the sources:

```
git clone git@github.com:sdot-team/sdot.git
cd sdot
pip install -e .
```

## Framework

SDOT can work with a lot of different frameworks. Here are the links for the official installation instructions for some of them (one is enough for most of the applications :) ):

- [JAX](https://docs.jax.dev/en/latest/installation.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [NumPy](https://numpy.org/install/)

> **GPU support** — follow the GPU installation instructions of the chosen framework for your platform (CUDA, …). SDOT will automatically use the devices your tensors are on, but some of the frameworks need specific installation instructions.

## Compilers

The kernels of the common cases come precompiled with the package. A more "exotic" case is compiled on first use, which
needs `ninja`, a C++ compiler, and `nvcc` for NVIDIA GPUs.

## Hello, World

Here is a simple program to test if the installation works: the Wasserstein distance between 100 diracs and an image.

::: code-group

<<< @/snippets/hello_numpy.py [NumPy]

<<< @/snippets/hello_jax.py [JAX]

<<< @/snippets/hello_torch.py [PyTorch]

:::

`sdot.plan( f, g )` gives the whole transport plan, and `p.show()` displays it ( inline in a notebook, in the browser
otherwise ):

<script setup>
import { withBase } from 'vitepress'
</script>

<iframe
  :src="withBase('/examples/SdotPlanNd-start.html') + '?wheel=0&touch=0'"
  title="SdotPlanNd interactive viewer"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 18vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0;"
></iframe>

<p style="text-align: right; margin: 0"><a :href="withBase('/examples/SdotPlanNd-start.html')" target="_blank">Open in a separate page ↗</a></p>

## What's Next

- [W2, plans and settings →](./ot-plan): `w2`, `plan`, `ot_solve`
- [Distributions →](./distributions): the sources and the targets
- [Differentiability →](./differentiability)
- [Frameworks and devices →](./backends)
- [Background →](./background): the mathematics
- [Examples gallery →](/examples/)
