#pragma once

// =====================================================================================
// THE LAPLACIAN OF THE LAGUERRE GRAPH, assembled from the facets ( `solvers_des_familles`,
// `src/solver/Laplacian.h`, taken over as is ).
//
//     c_ij = int_{facet ij} rho / ( 2 |p_i - p_j| ),   L_ii = sum_j c_ij,   L_ij = -c_ij
//
// The formula does not depend on the dimension: length over distance in 2D, area over distance in
// 3D, and it is the cell that delivers the facet's measure. Its kernel is exactly the
// CONSTANTS -- adding the same constant to all weights changes no cell.
//
// WITHOUT A SINGLE SORT. Each facet is seen TWICE, once per cell, and the two measures only
// differ by rounding; we keep the one of the row, so a count and a prefix sum
// place everyone -- and each row sums EXACTLY to zero. Pairing the two measures with
// a sort cost more than the diagram itself ( 4.2 s against 3.9 at n=1e6 ).
//
// THE GAUGE `w_0 = 0`: the reduced system strikes out row and column 0, and becomes positive definite.
// `reduced_crs` delivers it that way, columns sorted, for a solver that wants CRS.
// =====================================================================================

#include <loom/support/common_types.h>
#include <loom/support/kernels/CpuThreadPool.h>
#include <algorithm>
#include <vector>

namespace sdot {
namespace sdotplan {

/// A facet seen FROM cell `i`: it is the one that measured its extent.
struct Facet {
    SI     i, j;
    double c;                      ///< `c_ij > 0`; the minus sign is in the matrix
};

struct Laplacian {
    SI                  n = 0;
    std::vector<SI>     row, col;  ///< CSR of the off-diagonals, `c[ k ] = c_ij`
    std::vector<double> c;
    std::vector<double> dia;       ///< `L_ii`, the sum of the row

    struct Run { SI b, e, i; };      ///< facets `[ b, e )` of row `i`
    std::vector<std::vector<Run>> runs;   ///< ( scratch of `assemble`, per slice )
    std::vector<SI>     at;

    /// THE PARALLEL ASSEMBLY: the facets come cell by cell ( each sweep thread emits its cells' facets in a row ), so they are
    /// cut into RUNS of the same row, found in parallel slices; the runs are placed in the order of the facets ( two short
    /// sequential passes, one entry per run ), copied in parallel, and the diagonals summed in parallel in the order of the
    /// row. The result is the one of the sequential assembly, bit for bit, whatever the number of threads.
    void assemble( SI nb, const std::vector<Facet> &fa ) {
        n = nb;
        const SI nf = SI( fa.size() );
        const int T = nf < 65536 ? 1 : std::max( cpu_thread_pool().nb_workers(), 1 );
        auto run = [&]( auto &&slice ) { if ( T == 1 ) slice( 0 ); else cpu_thread_pool().run_threads( T, slice ); };

        runs.resize( T );
        run( [&]( int t ) {
            auto &rs = runs[ t ];
            rs.clear();
            const SI e = nf * ( t + 1 ) / T;
            for ( SI k = nf * t / T; k < e; ) {
                const SI b = k, i = fa[ k ].i;
                while ( ++k < e && fa[ k ].i == i ) {}
                rs.push_back( Run{ b, k, i } );
            }
        } );

        row.assign( n + 1, 0 );
        for ( const auto &rs : runs )
            for ( const Run &r : rs )
                row[ r.i + 1 ] += r.e - r.b;
        for ( SI i = 0; i < n; ++i )
            row[ i + 1 ] += row[ i ];
        at.assign( row.begin(), row.end() - 1 );
        for ( auto &rs : runs )
            for ( Run &r : rs ) {
                const SI dst = at[ r.i ];
                at[ r.i ] += r.e - r.b;
                r.i = dst;                               // from here on, `i` is where the run goes
            }

        col.resize( row[ n ] );
        c.resize( row[ n ] );
        dia.resize( n );
        run( [&]( int t ) {
            for ( const Run &r : runs[ t ] )
                for ( SI k = r.b, p = r.i; k < r.e; ++k, ++p ) {
                    col[ p ] = fa[ k ].j;
                    c[ p ] = fa[ k ].c;
                }
        } );
        symmetrize( T );
        // a cell without a neighbor cannot happen as long as none is empty, but a zero row
        // would make the system singular WITHOUT SAYING SO: we neutralize it.
        run( [&]( int t ) {
            const SI e = n * ( t + 1 ) / T;
            for ( SI i = n * t / T; i < e; ++i ) {
                double s = 0;
                for ( SI k = row[ i ]; k < row[ i + 1 ]; ++k )
                    s += c[ k ];
                dia[ i ] = s > 0 ? s : 1;
            }
        } );
    }

    /// THE TWO MEASURES OF A FACET, made one: `c_ij = c_ji = ( c_ij + c_ji ) / 2`, and a facet that only one of its cells saw is
    /// dropped. They differ by more than rounding where the facet is degenerate: a facet of zero length that one cell keeps and
    /// the other does not ( four seeds on a circle, every vertex of a grid ), or a facet lying ON a discontinuity of the density,
    /// which each cell integrates with the density of its own side ( a grid of seeds on an image: 5 % apart ). The solvers
    /// ( CG, the multigrid ) need a symmetric matrix: on a grid against an image, the multigrid stalled at 0.2 after 2500
    /// iterations. A lookup in the other row, a few entries long: no sort. Several entries of one row may share a column ( the
    /// facets of a group sent onto one row ): the SUMS `S_ij` and `S_ji` are made equal, each entry scaled by the same factor.
    std::vector<double> cs;
    void symmetrize( int T ) {
        cs.resize( c.size() );
        auto run = [&]( auto &&slice ) { if ( T == 1 ) slice( 0 ); else cpu_thread_pool().run_threads( T, slice ); };
        run( [&]( int t ) {
            const SI e = n * ( t + 1 ) / T;
            for ( SI i = n * t / T; i < e; ++i )
                for ( SI k = row[ i ]; k < row[ i + 1 ]; ++k ) {
                    const SI j = col[ k ];
                    if ( j < 0 || j >= n ) { cs[ k ] = c[ k ]; continue; }
                    double sij = 0, sji = 0;
                    for ( SI l = row[ i ]; l < row[ i + 1 ]; ++l )
                        if ( col[ l ] == j ) sij += c[ l ];
                    for ( SI l = row[ j ]; l < row[ j + 1 ]; ++l )
                        if ( col[ l ] == i ) sji += c[ l ];
                    cs[ k ] = sji > 0 && sij > 0 ? c[ k ] * ( 0.5 * ( sij + sji ) / sij ) : 0.0;
                }
        } );
        c.swap( cs );
    }

    /// `y = L x` ( the full system, gauge included )
    void product( const double *x, double *y ) const {
        for ( SI i = 0; i < n; ++i ) {
            double s = dia[ i ] * x[ i ];
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                s -= c[ e ] * x[ col[ e ] ];
            y[ i ] = s;
        }
    }

    /// THE REDUCED SYSTEM ( `n - 1` unknowns, seed 0 struck out -- not neutralized: an identity row
    /// would give the aggregation an isolated node to deal with ), diagonal included, columns
    /// SORTED by insertion -- seven entries per row.
    void reduced_crs( std::vector<int> &ptr, std::vector<int> &cl, std::vector<double> &val ) const {
        const SI m = n - 1;
        ptr.assign( m + 1, 0 );
        for ( SI i = 1; i < n; ++i ) {
            int k = 1;                                  // the diagonal, always present
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                k += col[ e ] >= 1;
            ptr[ i ] = k;
        }
        for ( SI i = 0; i < m; ++i )
            ptr[ i + 1 ] += ptr[ i ];
        cl.resize( ptr[ m ] );
        val.resize( ptr[ m ] );
        for ( SI i = 1; i < n; ++i ) {
            int k = ptr[ i - 1 ];
            cl[ k ] = int( i - 1 );  val[ k ] = dia[ i ];  ++k;
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                if ( col[ e ] >= 1 ) {
                    cl[ k ] = int( col[ e ] - 1 );  val[ k ] = -c[ e ];  ++k;
                }
            for ( int u = ptr[ i - 1 ] + 1; u < k; ++u ) {
                const int cc = cl[ u ];
                const double v = val[ u ];
                int j = u;
                for ( ; j > ptr[ i - 1 ] && cl[ j - 1 ] > cc; --j ) {
                    cl[ j ] = cl[ j - 1 ]; val[ j ] = val[ j - 1 ];
                }
                cl[ j ] = cc; val[ j ] = v;
            }
        }
    }
};

} // namespace sdotplan
} // namespace sdot
