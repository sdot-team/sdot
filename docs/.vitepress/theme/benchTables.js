// The tables of the performance page, read from the files of `bench/page.py` ( `guide/bench/<machine>.json` ).
//
// A table has rows and columns; a cell is the run of a machine whose job and params match both the row's `m` and the
// column's `m` ( `kind`: where it computed, cpu or gpu ). Columns repeat for every machine that has data for them.
// `v` reads the value from the run's results, `f` formats it; `calc( get )` computes a cell from other runs instead.

const ns   = x => x == null || x < 0 ? null : x >= 100 ? Math.round( x ).toLocaleString( 'en' ).replace( /,/g, ' ' ) : x.toPrecision( 2 )
const sec  = x => x == null ? null : x >= 100 ? `${ Math.round( x ) } s` : `${ x.toPrecision( 3 ) } s`
const mus  = x => x == null ? null : x >= 1000 ? `${ ( x / 1000 ).toPrecision( 2 ) } ms` : `${ x.toPrecision( 2 ) } µs`
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

// the columns of one program on the CPU: 1 thread, 8 threads and all the cores ( `page.py --threads`; a column with no run is
// left out, as the 8 threads of a machine of 8 cores, whose runs are those of all the cores )
const threadCols = ( name, m, v, f ) => [
  { label: `${ name }, 1 thread`, kind: 'cpu', m: { ...m, threads: 1 }, v, f },
  { label: `${ name }, 8 threads`, kind: 'cpu', m: { ...m, threads: 8 }, v, f },
  { label: r => `${ name }, ${ thr( r ) ?? 'all' } threads`, kind: 'cpu', m: { ...m, threads: -1 }, v, f },
]
const nsSeed = r => r.results.ns_per_seed
const triSeed = r => r.results.t_triangulation / r.results.n * 1e9
const solve = x => x && ( x.converged ? sec( x.t_total_wall ) : `no conv. ( ${ sec( x.t_total_wall ) } )` )

export const TABLES = {
  diagram: {
    note: 'ns per dirac, the measure of every cell, double ( GPU: kernel time ). CGAL: triangulation ( in parallel in 3D only ), then each cell rebuilt and clipped ( over the threads ). geogram: triangulation of the lifted seeds, then the cells restricted to a mesh of the box.',
    rows: CLOUDS,
    cols: [
      { label: 'n', kind: 'cpu', m: { job: 'diagram', threads: 1 }, v: nOf, f: x => x?.toExponential( 0 ).replace( 'e+', 'e' ) },
      ...threadCols( 'sdot', { job: 'diagram' }, nsSeed, ns ),
      { label: 'sdot, GPU', kind: 'gpu', m: { job: 'gpu_diagram', kernel: 'double' }, v: r => r.results.ns_per_seed_kernel, f: ns },
      ...threadCols( 'CGAL', { job: 'cgal', output: 'measures' }, nsSeed, ns ),
      ...threadCols( 'geogram', { job: 'geogram' }, nsSeed, ns ),
    ],
  },
  dual: {
    note: 'ns per dirac, the exact dual: the regular triangulation, with exact predicates and a symbolic perturbation of the ties. sdot: `dual()`, the cells cut with canonical predicates, each simplex written by the cell of its smallest seed ( restricted to the box ); GPU: kernel time, double. CGAL and geogram: their regular triangulation of the whole space ( CGAL in parallel in 3D only; geogram\'s 2D one is sequential ).',
    rows: CLOUDS,
    cols: [
      ...threadCols( 'sdot', { job: 'dual' }, nsSeed, ns ),
      { label: 'sdot, GPU', kind: 'gpu', m: { job: 'gpu_dual' }, v: r => r.results.ns_per_seed_kernel, f: ns },
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
      { label: 'preparation', kind: 'cpu', m: { job: 'sequence', threads: 1 }, v: r => r.results.t_prepare / r.results.n * 1e9, f: ns },
      ...threadCols( 'next diagram', { job: 'sequence' }, r => r.results.ns_per_seed_next, ns ),
    ],
  },
  newton: {
    note: 'A whole solve, cold start, tree included, `max |mass - target| / target <= 1e-6`. CPU: all the physical cores, double. GPU: jitted, `mixed`.',
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
      { label: r => `CPU, ${ thr( r ) ?? 'all' } threads`, kind: 'cpu', m: { job: 'newton', threads: -1 }, v: r => r.results.t_total_wall, f: sec },
      { label: 'GPU', kind: 'gpu', m: { job: 'gpu_newton' }, v: r => r.results.t_jit_wall ?? r.results.t_total_wall, f: sec },
      { label: 'speed-up', calc: get => ratio( get( 'cpu', { job: 'newton', threads: -1 } )?.results?.t_total_wall,
                                              get( 'gpu', { job: 'gpu_newton' } )?.results?.t_jit_wall ) },
      { label: 'iterations / diagrams', kind: 'cpu', m: { job: 'newton', threads: -1 }, v: r => r.results, f: x => x && `${ x.iterations } / ${ x.diagrams }` },
    ],
  },
  ot_geogram: {
    note: 'A whole semi-discrete solve, Lebesgue on the unit box, n = 1e5, cold start ( zero weights ), double ( GPU: jitted, `mixed` ), stopped on `max |mass - target| / target <= 1e-6` ( geogram\'s residual measured again by sdot ).',
    rows: [
      { label: '2D uniform',               m: { case: 'uniform', dim: 2, n: 0 }, gpu: { n: 100000 } },
      { label: '2D clustered ( 5 lines )', m: { case: 'lines_voronoi' } },
      { label: '3D uniform',               m: { case: 'uniform', dim: 3, n: 0 }, gpu: { n: 100000 } },
      { label: '3D clustered ( 4 planes )', m: { case: 'planes_voronoi' } },
    ],
    cols: [
      ...threadCols( 'sdot', { job: 'newton' }, r => r.results.t_total_wall, sec ),
      { label: 'sdot, GPU', kind: 'gpu', m: { job: 'gpu_newton' }, v: r => r.results.t_jit_wall ?? r.results.t_total_wall, f: sec },
      { label: 'iterations / diagrams', kind: 'cpu', m: { job: 'newton', threads: -1 }, v: r => r.results, f: x => x && `${ x.iterations } / ${ x.diagrams }` },
      ...threadCols( 'geogram', { job: 'geogram_newton' }, r => r.results, solve ),
      { label: 'iterations / diagrams', kind: 'cpu', m: { job: 'geogram_newton', threads: -1 }, v: r => r.results, f: x => x && `${ x.iterations } / ${ x.diagrams }` },
      { label: 'linear solves', kind: 'cpu', m: { job: 'geogram_newton', threads: -1 }, v: r => r.results.t_linear, f: sec },
      { label: 'geogram / sdot', calc: get => { const g = get( 'cpu', { job: 'geogram_newton', threads: -1 } )?.results
                                               return g?.converged ? ratio( g.t_total_wall, get( 'cpu', { job: 'newton', threads: -1 } )?.results?.t_total_wall ) : null } },
    ],
  },
  gpu_diagram: {
    note: 'ns per dirac, kernel time only.',
    rows: CLOUDS,
    cols: [
      { label: 'float',  kind: 'gpu', m: { job: 'gpu_diagram', kernel: 'float' },  v: r => r.results.ns_per_seed_kernel, f: ns },
      { label: 'double', kind: 'gpu', m: { job: 'gpu_diagram', kernel: 'double' }, v: r => r.results.ns_per_seed_kernel, f: ns },
    ],
  },
  scaling: {
    note: 'A whole solve on the card, `mixed`, uniform diracs, Lebesgue. 1e7: eager ( `--full` ).',
    rows: [ 2, 3 ].flatMap( d => [ 1e5, 1e6, 1e7 ].map( n => ( { label: `${ d }D, ${ n.toExponential( 0 ).replace( 'e+', 'e' ) }`, m: { case: 'uniform', dim: d, n } } ) ) ),
    cols: [
      { label: 'total', kind: 'gpu', m: { job: [ 'gpu_newton', 'gpu_n1e7' ] }, v: r => r.results.t_jit_wall ?? r.results.t_eager_wall ?? r.results.t_total_wall, f: sec },
      { label: 'per dirac', kind: 'gpu', m: { job: [ 'gpu_newton', 'gpu_n1e7' ] }, v: r => ( r.results.t_jit_wall ?? r.results.t_eager_wall ?? r.results.t_total_wall ) / r.results.n * 1e6, f: mus },
    ],
  },
  dims: {
    note: 'Lebesgue on the unit hypercube, n = 1e4, cold start, all the physical cores, double, one run. 6D: `--full`.',
    rows: [ 2, 3, 4, 5, 6 ].map( d => ( { label: `${ d }D`, m: { dim: d } } ) ),
    cols: [
      { label: 'total', kind: 'cpu', m: { job: [ 'dims', 'dims6' ] }, v: r => r.results.t_total_wall, f: sec },
      { label: 'iterations / diagrams', kind: 'cpu', m: { job: [ 'dims', 'dims6' ] }, v: r => r.results, f: x => x && `${ x.iterations } / ${ x.diagrams }` },
      { label: 'one diagram, per dirac', kind: 'cpu', m: { job: [ 'dims', 'dims6' ] }, v: r => r.results.t_diag / r.results.diagrams / r.results.n * 1e6, f: mus },
    ],
  },
}
