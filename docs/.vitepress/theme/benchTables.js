// The tables of the performance page, read from the files of `bench/page.py` ( `guide/bench/<machine>.json` ).
//
// A table has rows and columns; a cell is the run of a machine whose job and params match both the row's `m` and the
// column's `m` ( `kind`: where it computed, cpu or gpu ). A `gpu` column repeats for every card of `SOURCES.gpu`, headed by
// the card's name; a `cpu` one reads the machine of `SOURCES.cpu`. `group` heads the consecutive columns that share it.
// `v` reads the value from the run's results, `f` formats it ( the row's when the column has none ); `calc( get, row )`
// computes a cell from other runs instead ( `get( 'cpu', m )` reads the CPU machine, `get( 'gpu', m )` the card of the
// column: `src` says which machines the column repeats over ). `best`: the cell holding the smallest value of its row
// among the `best` columns of the same kind is shaded ( `true`: the value itself; a function: the number to compare ).
// ONE COLUMN PER PROGRAM AND KIND: the thread counts of a program ( `merge: 'threads'` ) share a column, `a / b / c`, headed
// `1 / 32 / 128 threads`, and so do the cards of a `gpu` column ( `2080 Ti / H100` ); the shading marks the value, not the cell.

// the machines ( the `name` of their file ) each kind of column comes from. The CPU: a node of 2 x 64 cores, the furthest the
// threads go. The GPU: a consumer card and a cluster one, side by side.
export const SOURCES = { cpu: [ 'n14' ], gpu: [ 'lmo', 'jz-h100' ] }

const ns   = x => x == null || x < 0 ? null : x >= 100 ? Math.round( x ).toLocaleString( 'en' ).replace( /,/g, ' ' ) : x.toPrecision( 2 )
const sec  = x => x == null ? null : x >= 100 ? `${ Math.round( x ) } s` : `${ x.toPrecision( 3 ) } s`
const mus  = x => x == null ? null : x >= 1000 ? `${ ( x / 1000 ).toPrecision( 2 ) } ms` : x >= 100 ? `${ Math.round( x ) } µs` : `${ x.toPrecision( 2 ) } µs`
const ratio = ( a, b ) => a == null || b == null ? null : `${ ( a / b ).toPrecision( 2 ) }×`
const thr  = r => r?.results?.threads

const CLOUDS = [
  { label: '2D uniform, Voronoi',      m: { case: 'uniform', dim: 2, wscale: 0 } },
  { label: '2D uniform, Laguerre',     m: { case: 'uniform', dim: 2, wscale: 0.3 } },
  { label: '2D lines, Voronoi',        m: { case: 'lines_voronoi' } },
  { label: '2D lines, equal areas',    m: { case: 'lines_equal' } },
  { label: '3D uniform, Voronoi',      m: { case: 'uniform', dim: 3, wscale: 0 } },
  { label: '3D planes, Voronoi',       m: { case: 'planes_voronoi' } },
  { label: '3D planes, equal volumes', m: { case: 'planes_equal' } },
]
const nOf = r => r?.results?.n

// the columns of one program on the CPU: 1 thread, 8 threads and all the physical cores ( `page.py --threads`; a column with no
// run is left out, as the 8 threads of a machine of 8 cores, whose runs are those of all the cores )
const threadCols = ( group, m, v, f, best = true ) => [
  { group, label: '1', merge: 'threads', kind: 'cpu', m: { ...m, threads: 1 }, v, f, best },
  { group, label: '32', merge: 'threads', kind: 'cpu', m: { ...m, threads: 32 }, v, f, best },
  { group, label: r => `${ thr( r ) ?? 'all' }`, merge: 'threads', kind: 'cpu', m: { ...m, threads: 'cores' }, v, f, best },
]
const t1 = r => r.results.t_total_wall
const tGpu = r => r.results.t_jit_wall ?? r.results.t_total_wall
const nsSeed = r => r.results.ns_per_seed
const nsKernel = r => r.results.ns_per_seed_kernel
const triSeed = r => r.results.t_triangulation / r.results.n * 1e9
const solve = x => x && ( x.converged ? sec( x.t_total_wall ) : `no conv. ( ${ sec( x.t_total_wall ) } )` )
const its = x => x && `${ x.iterations } / ${ x.diagrams }`

export const TABLES = {
  diagram: {
    note: 'ns per dirac, the measure of every cell, double ( GPU: kernel time ). CGAL: triangulation ( in parallel in 3D only ), then each cell rebuilt and clipped ( over the threads ). geogram: triangulation of the lifted seeds, then the cells restricted to a mesh of the box. Shaded: the fastest on the CPU, and on the GPU.',
    rows: CLOUDS,
    cols: [
      { label: 'n', kind: 'cpu', m: { job: 'diagram', threads: 1 }, v: nOf, f: x => x?.toExponential( 0 ).replace( 'e+', 'e' ) },
      ...threadCols( 'sdot', { job: 'diagram' }, nsSeed, ns ),
      { group: 'sdot', kind: 'gpu', m: { job: 'gpu_diagram', kernel: 'double' }, v: nsKernel, f: ns, best: true },
      ...threadCols( 'CGAL', { job: 'cgal', output: 'measures' }, nsSeed, ns ),
      ...threadCols( 'geogram', { job: 'geogram' }, nsSeed, ns ),
    ],
  },
  dual: {
    note: 'ns per dirac, the exact dual: the regular triangulation, with exact predicates and a symbolic perturbation of the ties. sdot: `dual()`, the cells cut with canonical predicates, each simplex written by the cell of its smallest seed ( restricted to the box ); GPU: kernel time, double. CGAL and geogram: their regular triangulation of the whole space ( CGAL in parallel in 3D only; geogram\'s 2D one is sequential ). Shaded: the fastest on the CPU, and on the GPU.',
    rows: CLOUDS,
    cols: [
      ...threadCols( 'sdot', { job: 'dual' }, nsSeed, ns ),
      { group: 'sdot', kind: 'gpu', m: { job: 'gpu_dual' }, v: nsKernel, f: ns, best: true },
      ...threadCols( 'CGAL', { job: 'cgal_dual' }, nsSeed, ns ),
      ...threadCols( 'geogram', { job: 'geogram' }, triSeed, ns ),
    ],
  },
  sequence: {
    note: 'ns per dirac. 10 diagrams on the same positions, the weights going from 0 to a Laguerre diagram: the preparation ( tree, structures ) once, then each diagram reuses it.',
    rows: [
      { label: '2D uniform, 1e6', m: { dim: 2 } },
      { label: '3D uniform, 1e6', m: { dim: 3 } },
    ],
    cols: [
      ...threadCols( 'preparation', { job: 'sequence' }, r => r.results.t_prepare / r.results.n * 1e9, ns, false ),
      ...threadCols( 'next diagram', { job: 'sequence' }, r => r.results.ns_per_seed_next, ns, false ),
    ],
  },
  newton: {
    note: 'A whole solve, cold start, tree included, `max |mass - target| / target <= 1e-6`. CPU: all the physical cores, double. GPU: jitted, `mixed`; speed-up: against the CPU.',
    rows: [
      { label: '2D uniform, 1e5',            m: { case: 'uniform', dim: 2, n: 100000 }, cpu: { n: 0 } },
      { label: '2D uniform, 1e6',            m: { case: 'uniform', dim: 2, n: 1000000 }, cpu: null },
      { label: '2D clustered ( 5 lines ), 1e5', m: { case: 'lines_voronoi' } },
      { label: '2D 512² image, 2e4',         m: { case: 'image' } },
      { label: '2D 4 gaussians ( σ = 0.02 ), 1e5', m: { case: 'gauss4' } },
      { label: '3D uniform, 1e5',            m: { case: 'uniform', dim: 3, n: 100000 }, cpu: { n: 0 } },
      { label: '3D uniform, 1e6',            m: { case: 'uniform', dim: 3, n: 1000000 }, cpu: null },
      { label: '3D clustered ( 4 planes ), 1e5', m: { case: 'planes_voronoi' } },
    ],
    cols: [
      { group: 'CPU', label: r => `${ thr( r ) ?? 'all' } threads`, kind: 'cpu', m: { job: 'newton', threads: 'cores' }, v: t1, f: sec },
      { group: 'CPU', label: 'iterations / diagrams', kind: 'cpu', m: { job: 'newton', threads: 'cores' }, v: r => r.results, f: its },
      { group: 'GPU', kind: 'gpu', m: { job: 'gpu_newton' }, v: tGpu, f: sec },
      { group: 'speed-up', src: 'gpu', calc: get => { const c = get( 'cpu', { job: 'newton', threads: 'cores' } ), g = get( 'gpu', { job: 'gpu_newton' } )
                                                      return c && g && ratio( t1( c ), tGpu( g ) ) } },
    ],
  },
  ot_geogram: {
    note: 'A whole semi-discrete solve, Lebesgue on the unit box, n = 1e5, cold start ( zero weights ), double ( GPU: jitted, `mixed` ), stopped on `max |mass - target| / target <= 1e-6` ( geogram\'s residual measured again by sdot ). Shaded: the fastest on the CPU, and on the GPU.',
    rows: [
      { label: '2D uniform',               m: { case: 'uniform', dim: 2, n: 0 }, gpu: { n: 100000 } },
      { label: '2D clustered ( 5 lines )', m: { case: 'lines_voronoi' } },
      { label: '3D uniform',               m: { case: 'uniform', dim: 3, n: 0 }, gpu: { n: 100000 } },
      { label: '3D clustered ( 4 planes )', m: { case: 'planes_voronoi' } },
    ],
    cols: [
      ...threadCols( 'sdot', { job: 'newton' }, t1, sec ),
      { group: 'sdot', kind: 'gpu', m: { job: 'gpu_newton' }, v: tGpu, f: sec, best: true },
      { group: 'sdot', label: 'iterations / diagrams', kind: 'cpu', m: { job: 'newton', threads: 'cores' }, v: r => r.results, f: its },
      ...threadCols( 'geogram', { job: 'geogram_newton' }, r => r.results, solve, x => x?.converged ? x.t_total_wall : null ),
      { group: 'geogram', label: 'iterations / diagrams', kind: 'cpu', m: { job: 'geogram_newton', threads: 'cores' }, v: r => r.results, f: its },
      { group: 'geogram', label: 'linear solves', kind: 'cpu', m: { job: 'geogram_newton', threads: 'cores' }, v: r => r.results.t_linear, f: sec },
    ],
  },
  gpu_diagram: {
    note: 'ns per dirac, kernel time only.',
    rows: CLOUDS,
    cols: [
      { group: 'float',  kind: 'gpu', m: { job: 'gpu_diagram', kernel: 'float' },  v: nsKernel, f: ns },
      { group: 'double', kind: 'gpu', m: { job: 'gpu_diagram', kernel: 'double' }, v: nsKernel, f: ns },
    ],
  },
  scaling: {
    note: 'A whole solve on the card, `mixed`, uniform diracs, Lebesgue. 1e7 ( `--full` ): an eager call, timed by its kernels.',
    rows: [ 2, 3 ].flatMap( d => [ 1e5, 1e6, 1e7 ].map( n => ( { label: `${ d }D, ${ n.toExponential( 0 ).replace( 'e+', 'e' ) }`, m: { case: 'uniform', dim: d, n } } ) ) ),
    cols: [
      { group: 'total', kind: 'gpu', m: { job: [ 'gpu_newton', 'gpu_n1e7' ] }, v: r => r.results.t_jit_wall ?? r.results.t_eager_kernels ?? r.results.t_total_wall, f: sec },
      { group: 'per dirac', kind: 'gpu', m: { job: [ 'gpu_newton', 'gpu_n1e7' ] }, v: r => ( r.results.t_jit_wall ?? r.results.t_eager_kernels ?? r.results.t_total_wall ) / r.results.n * 1e6, f: mus },
    ],
  },
  dims: {
    note: 'Lebesgue on the unit hypercube, n = 1e4, cold start, all the physical cores, double, one run. 6D: `--full`.',
    rows: [ 2, 3, 4, 5, 6 ].map( d => ( { label: `${ d }D`, m: { dim: d } } ) ),
    cols: [
      { label: 'total', kind: 'cpu', m: { job: [ 'dims', 'dims6' ] }, v: r => r.results.t_total_wall, f: sec },
      { label: 'iterations / diagrams', kind: 'cpu', m: { job: [ 'dims', 'dims6' ] }, v: r => r.results, f: its },
      { label: 'one diagram, per dirac', kind: 'cpu', m: { job: [ 'dims', 'dims6' ] }, v: r => r.results.t_diag / r.results.diagrams / r.results.n * 1e6, f: mus },
    ],
  },
}

// THE CHARTS: how each program scales, one panel per row ( the clouds of the tables ), log-log. A series is a program:
// its CPU runs ( `m`, every thread count found ) make a curve over the threads, and its runs on the cards ( `gpu.m` ) one
// point each, at the card's number of SMs. `v`: the time in ns ( per dirac or for the whole solve ) of a run, null when
// it does not count ( a solve that did not converge ). The charts plot CYCLES, the time times the clock the run measured
// ( `clock_ghz` on the CPU, `sm_clock_ghz` on the card ): a core alone runs faster than 128 busy ones, which times would
// take for a loss of scaling. Without a clock for every point, a chart falls back to the times.
const nsSolve = r => r.results.t_total_wall * 1e9
const nsSolveGpu = r => tGpu( r ) * 1e9
export const CHARTS = {
  diagram: {
    unit: 'per dirac',
    rows: CLOUDS,
    series: [
      // from scratch like the others ( the tree, the diagram, its first measures ), then prepared: the next diagrams on the
      // same diracs, which reuse the tree and the neighbours ( what a solver, a Lloyd step or a flow computes ); the cards: kernel time
      { label: 'sdot', m: { job: 'diagram' }, v: r => r.results.ns_per_seed_scratch,
        gpu: { m: { job: 'gpu_diagram', kernel: 'double' }, v: r => r.results.ns_per_seed_scratch_kernel } },
      { label: 'sdot, prepared', color: 0, dash: true, m: { job: 'diagram' }, v: nsSeed, gpu: { m: { job: 'gpu_diagram', kernel: 'double' }, v: nsKernel } },
      { label: 'CGAL', color: 1, m: { job: 'cgal', output: 'measures' }, v: nsSeed },
      { label: 'geogram', color: 2, m: { job: 'geogram' }, v: nsSeed },
    ],
  },
  dual: {
    unit: 'per dirac',
    rows: CLOUDS,
    series: [
      { label: 'sdot', m: { job: 'dual' }, v: nsSeed, gpu: { m: { job: 'gpu_dual' }, v: nsKernel } },
      { label: 'CGAL', m: { job: 'cgal_dual' }, v: nsSeed, flat: true },    // ( its 2D triangulation is sequential )
      { label: 'geogram', m: { job: 'geogram' }, v: triSeed },
    ],
  },
  ot_geogram: {
    unit: 'per solve',
    rows: [
      { label: '2D uniform, 1e5',               m: { case: 'uniform', dim: 2, n: 0 }, gpu: { n: 100000 } },
      { label: '2D clustered ( 5 lines ), 1e5', m: { case: 'lines_voronoi' } },
      { label: '3D uniform, 1e5',               m: { case: 'uniform', dim: 3, n: 0 }, gpu: { n: 100000 } },
      { label: '3D clustered ( 4 planes ), 1e5', m: { case: 'planes_voronoi' } },
    ],
    series: [
      { label: 'sdot', m: { job: 'newton' }, v: nsSolve, gpu: { m: { job: 'gpu_newton' }, v: nsSolveGpu } },
      { label: 'geogram', m: { job: 'geogram_newton' }, v: r => r.results.converged ? nsSolve( r ) : null },
    ],
  },
  // over the number of diracs ( `x: 'n'` ): the CPU on all its cores, each card a curve of its own, in time per dirac
  n_diagram: {
    x: 'n', unit: 'per dirac, the measure of every cell',
    rows: [ { label: '2D uniform', m: { case: 'uniform', dim: 2 } }, { label: '3D uniform', m: { case: 'uniform', dim: 3 } } ],
    series: [
      { label: 'sdot', m: { job: 'n_diagram', threads: 'cores' }, v: r => r.results.ns_per_seed_scratch,
        gpu: { m: { job: 'gpu_n_diagram' }, v: r => r.results.ns_per_seed_scratch_kernel } },
      { label: 'sdot, prepared', color: 0, dash: true, m: { job: 'n_diagram', threads: 'cores' }, v: nsSeed, gpu: { m: { job: 'gpu_n_diagram' }, v: nsKernel } },
      { label: 'CGAL, 32 threads ( its best )', color: 1, m: { job: 'n_cgal', threads: 32 }, v: nsSeed },
      { label: 'geogram', color: 2, m: { job: 'n_geogram', threads: 'cores' }, v: nsSeed },
    ],
  },
  n_newton: {
    x: 'n', unit: 'per dirac, a whole solve',
    rows: [ { label: '2D uniform', m: { case: 'uniform', dim: 2 } }, { label: '3D uniform', m: { case: 'uniform', dim: 3 } } ],
    series: [
      { label: 'sdot', m: { job: 'n_newton', threads: 'cores' }, v: r => nsSolve( r ) / r.results.n,
        gpu: { m: { job: [ 'gpu_newton', 'gpu_n1e7' ] }, // ( an eager run, the 1e7 ones, by its kernels: its wall time counts python's loop and any kernel compiled on the way )
        v: r => ( r.results.t_jit_wall ?? r.results.t_eager_kernels ?? r.results.t_total_wall ) * 1e9 / r.results.n } },
      { label: 'geogram', color: 2, m: { job: [ 'n_geogram_newton', 'n_geogram_newton_1e7' ], threads: 'cores' },
        v: r => r.results.converged ? nsSolve( r ) / r.results.n : null },
    ],
  },
  // over the dimension ( `x: 'dim'`, semi-log ): a panel per quantity ( `v` of the row: two units, two panels )
  dims: {
    x: 'dim', unit: 'n = 1e4, all the cores',
    rows: [ { label: 'A whole solve', v: r => r.results.t_total_wall * 1e9 },
            { label: 'One diagram, per dirac', v: r => r.results.t_diag / r.results.diagrams / r.results.n * 1e9 } ],
    series: [ { label: 'sdot', m: { job: [ 'dims', 'dims6' ] }, v: null } ],
  },
}
