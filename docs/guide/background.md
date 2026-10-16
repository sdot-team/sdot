# Background

This page builds intuition for what SDOT computes, how the solvers work, and what quantities you can extract from an OT plan.

## The base problem

Given:
- **f** — a discrete measure: a finite collection of weighted Dirac masses at positions $y_1, \ldots, y_n$ with masses $m_1, \ldots, m_n$ (with $\sum_i m_i = 1$)
- **g** — a continuous density on some domain $\Omega$

Find the **transport map** $T : \Omega \to \{y_1, \ldots, y_n\}$ that moves $g$ onto $f$ at minimum cost:

$$W_2^2(f, g) = \min_T \int_\Omega \|x - T(x)\|^2 \, g(x) \, dx \quad \text{s.t.} \quad \int_{T^{-1}(y_i)} g(x) \, dx = m_i \quad \forall i$$

---

There are variants for the ground metric, notably for periodicity (see [ground metrics](./ground_metrics)).

Non euclidean norms are also planed (see [roadmap](./roadmap))

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

The weights maximize a concave function, whose gradient is the vector of the mass errors: a damped Newton (+ some specific optimizations) method converges in a few iterations to a few dozen. The key cost is computing the power diagram and its integrals at each step, which runs in **O(n log n)** time.

> **Reference:** Kitagawa, Mérigot, Thibert —
> _Convergence of a Newton algorithm for semi-discrete optimal transport_, JEMS 2019.

## Computing a Plan

Functions like `sdot.w2` (for the 2 norm) give only the distance. If you want more information, you can use `sdot.plan` (with some default values)

```python
p = sdot.plan( f, g )

p.cost            # W2^2 (by default)
p.barycenters     # the barycenter of each cell, [ n, d ]
p.weights         # the weights w of the power diagram, [ n ]
p.power_diagram() # the geometry
```

See [plans and settings](./ot-plan).

## Barycenters and the Lloyd Algorithm

The **barycenters** `p.barycenters[ i ]` are the centroids of each transport cell:

$$b_i = \frac{1}{m_i} \int_{C_i(w)} x \, g(x) \, dx$$

Here is an example of use of the barycenter, in a **Lloyd algorithm** : moving each Dirac to its barycenter and repeating gives converges to an optimal quantization of $g$:

<<< @/snippets/ot_problem.py

Actually, if you are intersted in optimal quantization, `sdot.quantization` does it for you: see the [Lloyd quantization](/examples/lloyd-quantization) example (the snippet is here to show how to get the barycenters).

## What's Next

- [Examples gallery →](/examples/)
- [plans and settings →](./ot-plan)
