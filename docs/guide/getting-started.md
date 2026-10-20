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

SDOT can work with a lot of different frameworks. 

If you have not installed any (independently of SDOT), here are the links for the official installation instructions for some of them (one is enough):

- [JAX](https://docs.jax.dev/en/latest/installation.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [NumPy](https://numpy.org/install/)

> **GPU support** — follow the GPU installation instructions of the chosen framework for your platform (CUDA, …). SDOT will automatically use the devices your tensors are on, but some frameworks need specific installation instructions.

## Compilers

The kernels for the common cases come *precompiled with the package*. Less common cases are compiled on first use. For such a case, you will need `ninja`, a C++ compiler, and `nvcc` for NVIDIA GPUs.

## Hello, World

Here is a simple program to test if the installation works: the Wasserstein 2 distance between 100 diracs and an image.

::: code-group

<<< @/snippets/hello_numpy.py [NumPy]

<<< @/snippets/hello_jax.py [JAX]

<<< @/snippets/hello_torch.py [PyTorch]

<<< @/snippets/hello_cupy.py [CuPy]

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

- [Background →](./background) the mathematics
- [Examples gallery →](/examples/) some use cases
- [Plans and settings →](./ot-plan) barycenters, moments, ...
- [Distributions →](./distributions/) the sources and the targets
- [Frameworks and devices →](./backends) how sdot handles the frameworks
