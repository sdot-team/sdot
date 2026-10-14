# Background

This page builds intuition for what SDOT computes, how the Newton solver works, and what quantities you can extract from an OT plan.

---

## The Problem

Given:
- **f** — a discrete measure: a finite collection of weighted Dirac masses at positions $y_1, \ldots, y_n$ with masses $m_1, \ldots, m_n$ (with $\sum_i m_i = 1$)
- **g** — a continuous density on some domain $\Omega$

Find the **transport map** $T : \Omega \to \{y_1, \ldots, y_n\}$ that moves $g$ onto $f$ at minimum cost:

$$W_2^2(f, g) = \min_T \int_\Omega \|x - T(x)\|^2 \, g(x) \, dx \quad \text{s.t.} \quad \int_{T^{-1}(y_i)} g(x) \, dx = m_i \quad \forall i$$

---

## Power Diagrams (Laguerre Cells)

The optimal transport map in the semi-discrete setting is always a **power diagram** (also called a Laguerre tessellation): a partition of $\Omega$ into cells $C_i(w)$, each associated to a Dirac $y_i$, where membership is defined by a weighted distance:

$$x \in C_i(w) \iff \|x - y_i\|^2 - w_i \leq \|x - y_j\|^2 - w_j \quad \forall j \neq i$$

When all weights $w_i = 0$, this reduces to the standard Voronoi diagram. The weights "inflate" or "deflate" each cell to match the target mass.

<!-- INSERT IMAGE: power diagram with colored cells -->

The transport plan then simply sends every point $x$ in cell $C_i$ to position $y_i$.

---

## The Newton Solver

SDOT finds the optimal weights by solving:

$$\int_{C_i(w)} g(x) \, dx = m_i \quad \forall i$$

The weights maximize a concave function, whose gradient is the vector of the mass errors: a damped Newton method converges in a few iterations to a few dozen. The key cost is computing the power diagram and its integrals at each step, which runs in **O(n log n)** time.

> **Reference:** Kitagawa, Mérigot, Thibert —
> _Convergence of a Newton algorithm for semi-discrete optimal transport_, JEMS 2019.

---

## Computing a Plan

```python
p = sdot.plan( f, g )

p.cost          # W2^2
p.barycenters   # the barycenter of each cell, [ n, d ]
p.cell_masses   # the mass of each cell, [ n ]
p.weights       # the weights w of the power diagram, [ n ]
p.power_diagram()
```

See [W2, plans and settings](./ot-plan).

---

## The Wasserstein Distance

The cost `p.cost` equals:

$$W_2^2(f, g) = \int_\Omega \|x - T(x)\|^2 \, g(x) \, dx = \sum_i \int_{C_i} \|x - y_i\|^2 \, g(x) \, dx$$

`sdot.w2( f, g )` gives it without keeping the plan.

---

## Barycenters and the Lloyd Algorithm

The **barycenters** `p.barycenters[ i ]` are the centroids of each transport cell:

$$b_i = \frac{1}{m_i} \int_{C_i(w)} x \, g(x) \, dx$$

Moving each Dirac to its barycenter and repeating gives the **Lloyd algorithm**, which converges to an optimal quantization of $g$:

<<< @/snippets/ot_problem.py

`sdot.quantization` does it for you: see the [Lloyd quantization](/examples/lloyd-quantization) example.

---

## Gradients

At the optimal weights, the derivative of $W_2^2$ with respect to the position $y_i$ of a dirac is

$$\frac{\partial W_2^2}{\partial y_i} = 2 \, m_i \, (y_i - b_i)$$

It points from the barycenter toward the dirac, and vanishes exactly at the fixed points of Lloyd. JAX and PyTorch compute
it, and the other derivatives, automatically: see [Differentiability](./differentiability).

---

## What's Next

- [Example: Fitting a Gaussian mixture →](/examples/gaussian-mixture-fit)
- [Example: Lloyd quantization →](/examples/lloyd-quantization)
- [W2, plans and settings →](./ot-plan)
