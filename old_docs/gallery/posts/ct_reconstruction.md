---
categories: &id001
- PyTorch
- L-BFGS
- Wasserstein Distance
- 2D
date: '2026-03-16'
description: This example demonstrates how to reconstruct a 2D density from its projections
  using the semi-discrete Optimal Transport (W2 distance). It uses the L-BFGS optimizer
  to find the optimal coordinates of Dirac points.
hide:
- navigation
image: ../../assets/examples/ct_reconstruction/anim.gif
tags: *id001
title: Computed Tomography Reconstruction
---

This example demonstrates how to reconstruct a 2D density from its projections using the semi-discrete Optimal Transport (W2 distance). It uses the L-BFGS optimizer to find the optimal coordinates of Dirac points.

![Computed Tomography Reconstruction](../../assets/examples/ct_reconstruction/anim.gif){ align=center width=600 }

## Python Implementation

--8<-- "docs/examples/ct_reconstruction/ct_reconstruction.py"
