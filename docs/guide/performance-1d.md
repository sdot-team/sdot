# Performance: 1D Transport

The 1D plans of [1D Transport](./one-d), on batches of 256 independent problems: the threads of the host, or the card,
share them. Two kinds of measures only: diracs, and images of affine pixels ( `Image( degree = 1 )`, a piecewise linear
density ). Three cases:

1. [semi-discrete transports](#semi-discrete-transports): diracs against an image;
2. [transports between two images](#between-two-images), two continuous densities;
3. [against POT](#against-pot), on the transport between two images, discretized for POT.

The images sample sums of two thin gaussians ( standard deviation 0.05 ), shifted from one problem to the next. The diracs
are drawn uniformly. Everything is in double; the inputs are already where the computation runs, and the time stops when
the costs are on the host.

## Semi-discrete transports

`SdotPlan1d` on a batch: each problem's diracs sorted, then one sweep along its target, an image of 1e4 pixels. The costs
are exact.

<BenchChart chart="one_d" />

256 transports of 1e6 diracs take 0.82 s on 128 cores, 0.09 s on an H100.

## Between two images

`ContinuousOtPlan1d` between two images: no sort, the merge of the pixels of both, O( pixels ).

<BenchChart chart="one_d_images" />

256 pairs of images of 1e6 pixels take 1.6 s on 128 cores, 1.7 s on the H100.

The cost is exact up to the quadrature of each merged piece: against an affine target, the transport map involves a
square root, and so does its integral ( a closed form with an arcsinh, unstable when a slope vanishes ). Seven
Gauss-Legendre points per piece agree with a 64-point reference to 7e-15 on images of 1024 pixels.

## Over the threads

The problems are independent: the threads share them.

<BenchChart chart="one_d_threads" />

The images gain 13 times on 128 cores, the diracs 11 times; past 32 threads both gain little, bound by the memory.

## Against POT

POT has no continuous transport. The fair comparison is therefore on two continuous densities, both discretized alike
for POT: each image becomes `m` diracs, the centres of a regular grid of its support weighted by the masses of their
cells, the same number on both sides, then `ot.wasserstein_1d( p = 2 )`, one column per pair. The more diracs, the closer
POT gets to the continuous $W_2$, which sdot computes exactly: the chart puts the time of each method against the error
on $W_2$. The two do not solve the same problem; the chart gives the time POT needs for a given accuracy, at
as many threads: sdot on one thread against POT on numpy, on all the cores against POT on torch, and on the card against
POT on its jax backend.

<BenchChart chart="one_d_pot" />

POT's error on $W_2$ decreases as 1 / m² ( two digits per decade of diracs ). sdot, exact, takes 0.047 s on 128 cores:
POT on torch, on the same cores, takes as long for an error of about 1e-6, 0.31 s for 2.4e-8 and 25 s for 2.7e-12. On one
thread, 0.32 s for sdot against 0.6 s for POT on numpy at 2.4e-8. On the H100, POT is faster down to about 1e-9 ( 7 ms
for 2.4e-8, 34 ms for 2.6e-9, against 40 ms for sdot, of which 18 ms of kernels: the rest is the cost of the calls ),
and slower beyond ( 1.6 s for 2.6e-11 ); with 1e6 diracs per image, the 256 pairs no longer fit in its 80 GB.

## The tables

::: details Semi-discrete transports
<BenchTable table="one_d" />
:::

::: details Between two images
<BenchTable table="one_d_images" />
:::

::: details Over the threads
<BenchTable table="one_d_threads" />
:::

::: details Against POT
<BenchTable table="one_d_pot" />

<BenchTable table="one_d_pot_sdot" />
:::

The numbers come from [`bench/bench_1d.py`](https://github.com/sdot-team/sdot/tree/main/bench/bench_1d.py), jobs
`one_d*` of [`bench/page.py`](https://github.com/sdot-team/sdot/tree/main/bench/page.py). The machines and the protocol
are on the [overview](./performance#appendix-setup-and-protocol).
