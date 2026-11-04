# Performance: Power Diagrams

The cost of the diagram alone, for fixed weights ( the inner loop of every solver ), and of its exact dual.

## One power diagram

The cost of computing every cell's measure for fixed weights, the inner loop of every solver, per dirac. On the CPU, the
same clouds go through CGAL ( `Regular_triangulation_2/3`, bulk insertion ), which then rebuilds and clips every cell so
that both sides give the same result, and through geogram ( Lévy's `exploragram`: Delaunay triangulation of the lifted
seeds, then the cells restricted to a mesh of the box ), whose cell measures match sdot's to 1e-10.

sdot is timed twice. From scratch, like the others: the tree, the diagram and its first measures. Then PREPARED: a Newton
solve, a Lloyd step or a gradient flow at fixed positions computes many diagrams on the same diracs, and each new one
reuses the tree and the neighbours found by the previous one. CGAL and geogram have nothing to reuse: each new weight
vector costs their whole chain again. On the cards ( kernel time, double ), a diagram is a prepared one.

<BenchChart chart="diagram" />

On the 128 cores and a uniform cloud of 1e6 diracs, a 2D diagram costs 167 ns per dirac from scratch and 29 ns prepared;
a 3D one 332 ns and 110 ns. From scratch, the tree is two thirds of the 2D time on 128 cores: it gains 5× up to 32
threads, then slows down once the threads span both sockets. An H100 brings a prepared diagram down to 6 ns in 2D and 90 ns in 3D, and one from scratch to 8 ns and
92 ns: the card builds its tree in one call, for almost nothing. A 2080 Ti takes 57 ns and 590 ns prepared, 61 ns and
600 ns from scratch: its double units are few ( 13 ns and 270 ns in float ).
In float, the cell measures match the double kernel to a median relative error of about 1e-14 in 2D: the `mixed` mode
then finishes the solve in double.

### Scaling with the threads

A diagram is cut cell by cell, with no shared state. In 3D a prepared diagram gains 86× on 128 cores ( 114× in cycles ).
In 2D the cells are cheap and the memory bandwidth stops the gain past 96 cores, at 52× ( 67× in cycles ). On the
clustered 2D clouds the cells are uneven and the gain is lower ( 14× to 25× ). CGAL's 2D triangulation is sequential. Its
3D one runs in parallel ( TBB ) and gains 9× up to 64 threads, then slows down once the threads span both sockets: at
128 threads it is twice as slow as on one thread. On the clustered 3D clouds it stops gaining past 16 threads and then
does the same. geogram builds the cells in parallel, but its triangulation is sequential in 2D and gains about 3× in 3D,
and it is most of the cost.

### Scaling with the number of diracs

The CPU on all its cores, and each card, from 1e3 to 1e7 uniform diracs, in time per dirac: a flat curve is a cost linear
in `n`.

<BenchChart chart="n_diagram" />

On a card, a diagram of less than about 1e5 diracs does not fill the SMs, and below 1e4 the launches set its cost. Up
there, an H100 computes a prepared 2D diagram in 5 ns per dirac at 1e7 ( float ), and a 3D one in 72 ns; a 2080 Ti in
16 ns and 310 ns. From scratch, the tree adds 1 to 4 ns per dirac.

## Exact connectivity

sdot is not built for this: its cells are cut in floating point, and two cells may disagree about a vertex where several
seeds are almost on a sphere. `dual()` gives the regular triangulation restricted to the box, checked against qhull's on
random clouds. Where seeds are tied ( a grid, rings, seeds on a circle ) it gives one of the valid triangulations of the
tie.

In 2D, the connectivity is REPAIRED. Each cell is cut in floating point and only gives its list of neighbours. An edge is
kept when both of its seeds list it. When two kept edges cross ( the two diagonals of a square of a grid ), a single power
test removes one. The faces left are triangulated by one seed each. A triangle that its three cells propose is kept as it
is; a larger face is cut by the triangulation of its polygon that lies lowest on the lifted paraboloid. Each choice is made
once, so no two cells can disagree, and the floats only rank the choices. On random clouds almost every seed takes the
direct path, where its neighbours agree all around it. A tie of hidden seeds that leaves a face touching itself sends the
call to the certified path below.

In 3D each cell is cut in floating point, then CERTIFIED. A cell passes when no plane came within `1e-11` ( relative ) of
one of its vertices during the cut, and no vertex lies that close to a plane that is not its own. Its vertices are then
those of the exact cell. Only the doubtful cells are cut again with canonical predicates ( in double-double, with a
symbolic perturbation of the ties ). On random clouds, about one cell in ten thousand is cut again. On a grid, every cell
is. Double-double is not always enough: on a 2D grid of 10⁶ seeds nudged by 1e-14, a square is cocircular to 4e-35 and two
canonical cells take different diagonals ( the repair does not depend on such a precision ).

<BenchChart chart="dual" />

A triangulation is what CGAL and geogram are made for, and on one thread they stay ahead, in 3D especially: the
certification costs about a cell's worth on top of the cell itself. On the 128 cores sdot takes 160 ns per seed in 2D and
440 ns in 3D. On a grid, where every cell is cut canonically, it is ten times slower.

### Scaling with the threads

3D gains 45× on 128 cores ( 60× in cycles ), the clustered 3D clouds 28× and 44×. A uniform 2D cloud gains 11× and stops
there from 32 threads on: a 2D cell is too little work for its memory traffic. The clustered 2D clouds gain 5× and 9×.
CGAL's 2D triangulation is sequential: its time on one thread is drawn flat ( dashed ) over the threads.

## The tables

### One power diagram

<BenchTable table="diagram" />

On the GPU, each card in its own columns:

<BenchTable table="gpu_diagram" />

A sequence of diagrams on the same diracs: the preparation, then each next diagram.

<BenchTable table="sequence" />

### Exact connectivity

<BenchTable table="dual" />


The machines and the protocol ( cold starts, repetitions, clocks ) are on the [overview](./performance#appendix-setup-and-protocol).
