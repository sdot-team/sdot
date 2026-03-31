# Sdot

Welcome to the documentation for **Sdot**, a high-performance library for Semi-Discrete Optimal Transport.

Semi-Discrete means that **only one the the two measures has to be a sum of diracs**. It can be seen of a generalisation of discrete optimal transport (as can be seen in fantastic libraries like [geomloss](https://www.kernel-operations.io/geomloss/) or [pot](https://pythonot.github.io/)).

Roughly speaking, OT gives ways to compare two functions/measures while allowing "displacement", instead of just comparing the value for each point. [You can click here for a more detailed mathematical background](math_barkground.md)

## Features

- Works in any number of dimension
- Fast C++/Cuda implementation with Python bindings.
- Support for **PyTorch** and **JAX** (gradient for every inputs and outputs).

## Navigation

- [Examples](gallery/index.md)
- [API Reference](api.md)
