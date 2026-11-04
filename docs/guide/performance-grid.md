# Performance: Diracs on a Grid

The two paths of [Diracs on a Grid](./distributions/grid): the dedicated solver, and the same diracs given as a cloud
( a `SumOfDiracs` ), which is what a solve on positions does.

One problem, at every size: `N x N` diracs of equal masses on the unit square, against an image of 512 x 512 affine pixels
( `Image( degree = 1 )` ), a field of sines with an offset, `1 + 0.5 sin( 6 pi x ) sin( 4 pi y )`. Whole solves from a cold
start, stopped on `max |mass - target| / target <= 1e-6`; the CPU on all its cores ( 2 x 64 AMD EPYC 7763 ), the cards an
RTX 2080 Ti and an H100.

<BenchChart chart="grid" />

<BenchTable table="grid" />

The gap grows with `N`. On the CPU: 3× at 512 x 512, 10× at 2048 x 2048, 17× at 4096 x 4096 ( 3.2 s against 54 s on the
128 cores ). On the 2080 Ti: 5× at 512 x 512, 12× at 2048 x 2048 ( 0.35 s against 4.0 s ); on the H100, 30× at 2048 x 2048
( 0.072 s against 2.1 s ). Past a million diracs, a dirac costs the dedicated solver about 200 ns on the CPU, 83 ns on the
2080 Ti and 17 ns on the H100; the generic one more than 2 µs on the CPU, 0.5 to 1 µs on the cards. Below 256 x 256, fixed
costs dominate ( 0.03 to 0.15 s ).

The dedicated solver computes in doubles, on the CPU and on the card. The generic one computes in doubles on the CPU, and
on the card in floats first, then in doubles ( `precision = "auto"` ). Floats alone do not reach the tolerance: at 512 x
512 the generic solve stalls near 1e-3, the cells too small for the digits of a float.

The numbers on this page come from [`bench/bench_grid.py`](https://github.com/sdot-team/sdot/tree/main/bench/bench_grid.py)
( jobs `grid` and `gpu_grid` of `bench/page.py` ). The machines and the protocol of the other benchmarks are on the
[overview](./performance#appendix-setup-and-protocol).
