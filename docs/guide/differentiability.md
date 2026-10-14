# Differentiability

`sdot.w2` is differentiable with respect to everything that defines the two distributions: the positions and the masses
of the diracs, and the parameters of the target ( the pixels of an image, the centers, sigmas and masses of gaussians... ).
The outputs of a plan ( `cost`, `barycenters`, `cell_masses`, `weights` ) are differentiable too.

The solver itself is not differentiated: its derivative comes from the implicit function theorem, which costs one more
power diagram and one linear solve.

::: code-group

<<< @/snippets/grad_jax.py [JAX]

<<< @/snippets/grad_torch.py [PyTorch]

:::

`jax.jit` works too: the whole solve is one call.

## The gradient with respect to the positions

At the optimal weights, the derivative of $W_2^2$ with respect to the position $y_i$ of a dirac does not go through the
cells:

$$\frac{\partial W_2^2}{\partial y_i} = 2 \, m_i \, ( y_i - b_i )$$

where $m_i$ is the mass of the dirac and $b_i$ the barycenter of its cell:

<<< @/snippets/grad_formula.py

A gradient step of $1 / ( 2 m_i )$ therefore moves each dirac to its barycenter: that is the Lloyd algorithm.

The [Gaussian mixture fit](/examples/gaussian-mixture-fit) example fits the parameters of a `SumOfGaussians` by gradient
descent on `w2`.
