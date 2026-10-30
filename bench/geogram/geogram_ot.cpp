// GEOGRAM'S SEMI-DISCRETE OT ON A POINT CLOUD, TIMED: the driver of `bench_geogram.py`.
//
// geogram ( Bruno Levy ) computes power diagrams as the restriction of a (d+1)-dimensional Delaunay triangulation of the
// lifted seeds `( p, sqrt( W - w ) )` to a simplicial mesh of the domain ( `RestrictedVoronoiDiagram` ), and its
// semi-discrete transport ( exploragram, `OptimalTransportMap2d / 3d` ) is a KMT damped Newton on top of it. This program
// loads a cloud, meshes the unit square / cube ( `--grid` sub-squares / sub-cubes per side, 2 triangles / 6 tetrahedra
// each ), and runs one of
//
//   --mode diagram   the measure of every cell at the given weights: the regular triangulation ( `delaunay->set_vertices` )
//                    then the restricted cells ( `funcgrad`, which integrates over every cell ∩ simplex ). Min of `--reps`.
//   --mode adjacency the neighbours of every cell, with the facet terms ( `compute_P1_Laplacian`: the triangulation, then
//                    the restricted cells assembled in a sparse matrix, as sdot's `hessian_rows` ). Min of `--reps`.
//   --mode newton    `optimize_full_Newton` of exploragram, copied line for line with ONE change: it stops on
//                    `max |m_i - nu_i| / nu_i <= --rtol` ( sdot's rule ) instead of `|| m - nu ||_2 < sqrt( n ) eps nu`.
//                    `--multilevel 1`: Levy's multilevel scheme ( BRIO levels, ratio 0.125, weights of a level fitted
//                    on the previous one ), the coarse levels with geogram's own rule ( `--coarse-eps`, 0.01 like
//                    `compute_OTM` ), the last one with ours.
//
// The cloud is binary: int64 n, int64 d, n*d doubles ( positions ), n doubles ( weights, sdot's convention: the cell of
// i is where |x - p_i|^2 - w_i is the smallest -- geogram's too ). `--out` writes n doubles: the measures ( diagram ), the
// number of neighbours ( adjacency ) or the weights ( newton ), in the order of the input. Timings on stdout, one `key: value` per line, in seconds.
//
// Build: see COMPARISONS.md, "geogram".

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/process.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/permutation.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_reorder.h>
#include <exploragram/optimal_transport/optimal_transport_2d.h>
#include <exploragram/optimal_transport/optimal_transport_3d.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace GEO;

namespace {

double now() {
    return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}

// the unit square / cube cut in g^d sub-boxes, each in 2 triangles / 6 Kuhn tetrahedra ( conforming ), positively
// oriented; the vertices in dimension d + 1 ( the last coordinate 0 ), as exploragram wants them
void unit_box_mesh( Mesh& M, index_t d, index_t g ) {
    M.clear();
    M.vertices.set_dimension( d + 1 );
    index_t s = g + 1;
    auto vid = [&]( index_t i, index_t j, index_t k ) { return i + s * ( j + s * k ); };
    index_t nv = d == 2 ? s * s : s * s * s;
    M.vertices.create_vertices( nv );
    for( index_t k = 0; k < ( d == 2 ? 1 : s ); ++k )
        for( index_t j = 0; j < s; ++j )
            for( index_t i = 0; i < s; ++i ) {
                double* p = M.vertices.point_ptr( vid( i, j, k ) );
                p[ 0 ] = double( i ) / g; p[ 1 ] = double( j ) / g;
                if( d == 3 ) p[ 2 ] = double( k ) / g;
                p[ d ] = 0;
            }
    if( d == 2 ) {
        for( index_t j = 0; j < g; ++j )
            for( index_t i = 0; i < g; ++i ) {
                M.facets.create_triangle( vid( i, j, 0 ), vid( i + 1, j, 0 ), vid( i + 1, j + 1, 0 ) );
                M.facets.create_triangle( vid( i, j, 0 ), vid( i + 1, j + 1, 0 ), vid( i, j + 1, 0 ) );
            }
        M.facets.connect();
        return;
    }
    static const index_t perms[ 6 ][ 3 ] = { { 0, 1, 2 }, { 0, 2, 1 }, { 1, 0, 2 }, { 1, 2, 0 }, { 2, 0, 1 }, { 2, 1, 0 } };
    for( index_t k = 0; k < g; ++k )
        for( index_t j = 0; j < g; ++j )
            for( index_t i = 0; i < g; ++i )
                for( auto& pm : perms ) {
                    index_t c[ 3 ] = { i, j, k }, v[ 4 ];
                    v[ 0 ] = vid( c[ 0 ], c[ 1 ], c[ 2 ] );
                    for( index_t a = 0; a < 3; ++a ) {
                        ++c[ pm[ a ] ];
                        v[ a + 1 ] = vid( c[ 0 ], c[ 1 ], c[ 2 ] );
                    }
                    vec3 p0( M.vertices.point_ptr( v[ 0 ] ) ), p1( M.vertices.point_ptr( v[ 1 ] ) ),
                         p2( M.vertices.point_ptr( v[ 2 ] ) ), p3( M.vertices.point_ptr( v[ 3 ] ) );
                    if( Geom::tetra_signed_volume( p0, p1, p2, p3 ) < 0 )
                        std::swap( v[ 2 ], v[ 3 ] );
                    M.cells.create_tet( v[ 0 ], v[ 1 ], v[ 2 ], v[ 3 ] );
                }
    M.cells.connect();
    M.cells.compute_borders();
}

struct Stats {
    index_t iterations = 0, triangulations = 0, rvd = 0, trials = 0, cg_iters = 0;
    double t_tri = 0, t_rvd = 0, t_rvd_hessian = 0, t_lin = 0, residual = 0;
    bool converged = false;
    std::vector<std::string> trace;
};

// exposes the protected pieces of OptimalTransportMap ( geogram's own code, untouched )
template<class Base>
struct Driver : Base {
    Driver( Mesh* M, bool brio ) : Base( M, "default", brio ) {}

    using Base::points_dimp1_; using Base::dimp1_; using Base::dimension_; using Base::weights_; using Base::nb_air_particles_;
    using Base::delaunay_; using Base::w_did_not_change_; using Base::callback_; using Base::nbZ_; using Base::g_norm_;
    using Base::measure_of_smallest_cell_; using Base::linesearch_maxiter_; using Base::constant_nu_; using Base::total_mass_;
    using Base::level_;

    double nu_of( index_t i ) const { return this->nu( i ); }

    // the regular triangulation of the first n seeds at weights w ( the first half of `funcgrad` )
    void triangulate( index_t n, const double* w ) {
        double W = 0.0;
        for( index_t p = 0; p < n; ++p ) W = std::max( W, w[ p ] );
        for( index_t p = 0; p < n; ++p ) points_dimp1_[ dimp1_ * p + dimension_ ] = ::sqrt( W - w[ p ] );
        delaunay_->set_vertices( n + nb_air_particles_, points_dimp1_.data() );
    }

    // the restricted cells on the current triangulation: g = measures - nu ( + the Hessian if `hessian` )
    void cells( index_t n, double* w, double* g, bool hessian ) {
        double f = 0;
        w_did_not_change_ = true;
        if( hessian ) this->callback_->set_Newton_step( true );
        this->funcgrad( n, w, f, g );
        if( hessian ) this->callback_->set_Newton_step( false );
        w_did_not_change_ = false;
    }

    double max_rel( index_t n, const double* g ) const {
        double r = 0;
        for( index_t i = 0; i < n; ++i ) r = std::max( r, ::fabs( g[ i ] ) / nu_of( i ) );
        return r;
    }

    // `OptimalTransportMap::optimize_full_Newton`, the same steps; the stopping rule is sdot's ( see the header )
    void newton( index_t n, index_t max_iterations, double rtol, Stats& st ) {
        vector<double> pk( n ), xk( n ), gk( n );
        double epsilon0 = 0.0;
        auto tri = [&]( const double* w ) { double t0 = now(); triangulate( n, w ); st.t_tri += now() - t0; ++st.triangulations; };
        auto rvd = [&]( double* w, double* g, bool h ) { double t0 = now(); cells( n, w, g, h ); ( h ? st.t_rvd_hessian : st.t_rvd ) += now() - t0; ++st.rvd; };

        tri( weights_.data() );
        for( index_t k = 0; k < max_iterations; ++k ) {
            xk = weights_;
            this->new_linear_system( n, pk.data() );
            rvd( xk.data(), gk.data(), true );                                  // eval_func_grad_Hessian
            st.residual = max_rel( n, gk.data() );
            if( k == 0 ) {
                char buf[ 160 ];
                std::snprintf( buf, sizeof buf, "it 0 residual %.3e", st.residual );
                st.trace.push_back( buf );
            }
            if( st.residual <= rtol ) {                                         // already there ( the original always steps )
                nlEnd( NL_MATRIX ); nlEnd( NL_SYSTEM ); nlDeleteContext( nlGetCurrent() );
                st.converged = true;
                break;
            }
            if( epsilon0 == 0.0 ) {
                epsilon0 = measure_of_smallest_cell_;
                for( index_t i = 0; i < n; ++i ) epsilon0 = std::min( epsilon0, nu_of( i ) );
                epsilon0 = 0.5 * epsilon0;
            }
            if( nbZ_ != 0 ) {
                std::cerr << "geogram: empty cells, Newton stops ( as exploragram does )" << std::endl;
                nlEnd( NL_MATRIX ); nlEnd( NL_SYSTEM ); nlDeleteContext( nlGetCurrent() );
                break;
            }
            double t0 = now();
            nlEnd( NL_MATRIX );
            nlEnd( NL_SYSTEM );
            nlSolve();
            NLint used = 0;
            nlGetIntegerv( NL_USED_ITERATIONS, &used );
            st.cg_iters += index_t( used );
            nlDeleteContext( nlGetCurrent() );                                 // = solve_linear_system()
            st.t_lin += now() - t0;

            double alphak = 1.0, gknorm = g_norm_;
            index_t inner = 0;
            for( ; inner < linesearch_maxiter_; ++inner ) {
                for( index_t i = 0; i < n; ++i ) weights_[ i ] = xk[ i ] + alphak * pk[ i ];
                tri( weights_.data() );
                rvd( weights_.data(), gk.data(), false );
                ++st.trials;
                if( measure_of_smallest_cell_ >= epsilon0 && g_norm_ <= ( 1.0 - 0.5 * alphak ) * gknorm )
                    break;
                alphak /= 2.0;
            }
            st.residual = max_rel( n, gk.data() );
            ++st.iterations;
            char buf[ 200 ];
            std::snprintf( buf, sizeof buf, "it %u residual %.3e step %.3g trials %u cg %d", unsigned( k + 1 ), st.residual, alphak,
                           unsigned( inner + 1 ), int( used ) );
            st.trace.push_back( buf );
            if( st.residual <= rtol ) { st.converged = true; break; }
        }
    }
};

template<class OTM>
int run( index_t d, index_t n, std::vector<double>& pos, std::vector<double>& w, std::map<std::string, std::string>& a ) {
    auto arg = [&]( const char* k, const char* def ) { return a.count( k ) ? a[ k ] : std::string( def ); };
    std::string mode = arg( "mode", "diagram" );
    index_t grid = index_t( std::stoul( arg( "grid", "0" ) ) );
    index_t threads = Process::maximum_concurrent_threads();
    if( grid == 0 ) {                    // enough simplices for one part per thread, and not more ( each cut costs )
        grid = 1;
        while( ( d == 2 ? 2 * grid * grid : 6 * grid * grid * grid ) < 4 * threads ) ++grid;
    }
    bool multilevel = arg( "multilevel", "0" ) == "1";
    std::string out = arg( "out", "" );

    double t_setup0 = now();
    Mesh M;
    unit_box_mesh( M, d, grid );

    // multilevel: BRIO order, levels ( `sampling.cpp::BRIO_reorder`, `compute_OTM`'s ratio 0.125 )
    vector<index_t> order, levels;
    if( multilevel ) {
        compute_BRIO_order( n, pos.data(), order, coord_index_t( d ), d, 64, 0.125, &levels );
    } else {
        order.resize( n );
        for( index_t i = 0; i < n; ++i ) order[ i ] = i;
    }
    std::vector<double> P( size_t( n ) * d );
    for( index_t i = 0; i < n; ++i )
        for( index_t c = 0; c < d; ++c ) P[ size_t( i ) * d + c ] = pos[ size_t( order[ i ] ) * d + c ];

    Driver<OTM> ot( &M, multilevel );
    ot.set_verbose( false );
    ot.set_points( n, P.data() );
    if( a.count( "linsolve-eps" ) ) ot.set_linsolve_epsilon( std::stod( a[ "linsolve-eps" ] ) );
    if( a.count( "linsolve-maxiter" ) ) ot.set_linsolve_maxiter( index_t( std::stoul( a[ "linsolve-maxiter" ] ) ) );
    std::string solver = arg( "solver", "cg" );
    if( solver == "cholmod" ) ot.set_linear_solver( OT_CHOLMOD );
    if( solver == "superlu" ) ot.set_linear_solver( OT_SUPERLU );
    for( index_t i = 0; i < n; ++i ) ot.set_weight( i, w[ order[ i ] ] );
    double t_setup = now() - t_setup0;

    std::printf( "geogram: d %u n %u threads %u grid %u simplices %u total_mass %.17g setup %.6f s\n", unsigned( d ), unsigned( n ),
                 unsigned( threads ), unsigned( grid ), unsigned( d == 2 ? M.facets.nb() : M.cells.nb() ), ot.total_mass(), t_setup );

    std::vector<double> res( n );
    if( mode == "diagram" ) {
        index_t reps = index_t( std::stoul( arg( "reps", "3" ) ) );
        std::vector<double> g( n ), wt( ot.nb_points() );
        for( index_t i = 0; i < n; ++i ) wt[ i ] = ot.weight( i );
        double best_tri = 1e300, best_rvd = 1e300, best_tot = 1e300;
        for( index_t r = 0; r < reps + 1; ++r ) {                       // one warm-up round
            double t0 = now();
            ot.triangulate( n, wt.data() );
            double t1 = now();
            ot.cells( n, wt.data(), g.data(), false );
            double t2 = now();
            if( r == 0 ) continue;
            best_tri = std::min( best_tri, t1 - t0 );
            best_rvd = std::min( best_rvd, t2 - t1 );
            best_tot = std::min( best_tot, t2 - t0 );
        }
        double sum = 0, mn = 1e300;
        for( index_t i = 0; i < n; ++i ) {
            double m = g[ i ] + ot.nu_of( i );
            res[ order[ i ] ] = m;
            sum += m;
            mn = std::min( mn, m );
        }
        std::printf( "triangulation: %.6f s\nrvd: %.6f s\ndiagram: %.6f s\nsum_measures: %.17g\nmin_measure: %.6g\n",
                     best_tri, best_rvd, best_tot, sum, mn );
    } else if( mode == "adjacency" ) {
        // `compute_P1_Laplacian`: the triangulation, then the restricted cells with their facets, assembled in a sparse
        // matrix ( one row per seed: its neighbours and the facet terms ) -- what sdot's `hessian_rows` returns
        index_t reps = index_t( std::stoul( arg( "reps", "3" ) ) );
        std::vector<double> wt( n ), m( n );
        for( index_t i = 0; i < n; ++i ) wt[ i ] = ot.weight( i );
        double best = 1e300;
        for( index_t r = 0; r < reps + 1; ++r ) {                       // one warm-up round
            NLSparseMatrix H;
            nlSparseMatrixConstruct( &H, NLuint( n ), NLuint( n ), NL_MATRIX_STORE_ROWS );
            double t0 = now();
            ot.compute_P1_Laplacian( wt.data(), reinterpret_cast<NLMatrix>( &H ), m.data() );
            double t1 = now();
            if( r == reps ) {                                           // the neighbours of each seed, the diagonal apart
                for( index_t i = 0; i < n; ++i ) {
                    index_t nb = 0;
                    for( index_t k = 0; k < H.row[ i ].size; ++k )
                        nb += H.row[ i ].coeff[ k ].index != i;
                    res[ order[ i ] ] = double( nb );
                }
            }
            nlSparseMatrixDestroy( &H );
            if( r > 0 ) best = std::min( best, t1 - t0 );
        }
        double nnz = 0;
        for( double c : res ) nnz += c;
        std::printf( "adjacency: %.6f s\nmean_neighbours: %.6f\n", best, nnz / n );
    } else {
        index_t max_iter = index_t( std::stoul( arg( "max-iter", "100" ) ) );
        double rtol = std::stod( arg( "rtol", "1e-6" ) );
        Stats st;
        double t0 = now();
        if( multilevel ) {
            ot.set_epsilon( std::stod( arg( "coarse-eps", "0.01" ) ) );
            ot.set_Newton( true );
            for( index_t l = 0; l + 1 < levels.size(); ++l ) {          // `optimize_levels`
                ot.level_ = l + 1;
                vector<index_t> brio_levels;
                for( index_t i = 0; i <= l + 1; ++i ) brio_levels.push_back( levels[ i ] );
                ot.RVD()->delaunay()->set_BRIO_levels( brio_levels );
                bool last = l + 2 == levels.size();
                double tl = now();
                ot.optimize_level( levels[ l ], levels[ l + 1 ], last ? 0 : max_iter );   // last: only the fit of the weights
                if( !last )
                    std::printf( "level %u: %u seeds %.3f s\n", unsigned( l ), unsigned( levels[ l + 1 ] ), now() - tl );
            }
            st.t_tri = st.t_rvd = 0;                                    // the last level's own counts start here
            double tc = now() - t0;
            std::printf( "coarse_levels: %.6f s\n", tc );
        }
        ot.newton( n, max_iter, rtol, st );
        double t_newton = now() - t0;
        for( index_t i = 0; i < n; ++i ) res[ order[ i ] ] = ot.weight( i );
        for( auto& s : st.trace ) std::printf( "%s\n", s.c_str() );
        std::printf( "converged: %d\nresidual: %.6e\niterations: %u\ntriangulations: %u\nrvd_passes: %u\ntrials: %u\ncg_iterations: %u\n"
                     "t_triangulation: %.6f s\nt_rvd: %.6f s\nt_rvd_hessian: %.6f s\nt_linear: %.6f s\nt_newton: %.6f s\n",
                     int( st.converged ), st.residual, unsigned( st.iterations ), unsigned( st.triangulations ), unsigned( st.rvd ),
                     unsigned( st.trials ), unsigned( st.cg_iters ), st.t_tri, st.t_rvd, st.t_rvd_hessian, st.t_lin, t_newton );
    }
    if( !out.empty() ) {
        FILE* f = std::fopen( out.c_str(), "wb" );
        std::fwrite( res.data(), sizeof( double ), n, f );
        std::fclose( f );
    }
    return 0;
}

} // namespace

int main( int argc, char** argv ) {
    std::map<std::string, std::string> a;
    for( int i = 1; i + 1 < argc; i += 2 ) {
        if( std::strncmp( argv[ i ], "--", 2 ) != 0 ) { std::cerr << "bad argument " << argv[ i ] << std::endl; return 1; }
        a[ argv[ i ] + 2 ] = argv[ i + 1 ];
    }
    if( !a.count( "load" ) ) {
        std::cerr << "usage: geogram_ot --load cloud.bin [--mode diagram|adjacency|newton] [--threads k] [--grid g] [--reps r] [--out f]\n"
                     "                  [--rtol 1e-6] [--max-iter 100] [--multilevel 0|1] [--coarse-eps 0.01] [--solver cg|cholmod|superlu]\n"
                     "                  [--linsolve-eps 1e-3] [--linsolve-maxiter 1000]" << std::endl;
        return 1;
    }

    GEO::initialize( GEO::GEOGRAM_INSTALL_NONE );
    CmdLine::import_arg_group( "standard" );
    CmdLine::import_arg_group( "algo" );
    Logger::instance()->set_quiet( true );
    if( a.count( "threads" ) ) {
        index_t k = index_t( std::stoul( a[ "threads" ] ) );
        Process::enable_multithreading( k > 1 );
        Process::set_max_threads( k );
        omp_set_num_threads( int( k ) );                                // OpenNL's CG
    }

    std::ifstream f( a[ "load" ], std::ios::binary );
    std::int64_t hdr[ 2 ];
    f.read( reinterpret_cast<char*>( hdr ), sizeof hdr );
    index_t n = index_t( hdr[ 0 ] ), d = index_t( hdr[ 1 ] );
    std::vector<double> pos( size_t( n ) * d ), w( n );
    f.read( reinterpret_cast<char*>( pos.data() ), std::streamsize( pos.size() * sizeof( double ) ) );
    f.read( reinterpret_cast<char*>( w.data() ), std::streamsize( w.size() * sizeof( double ) ) );
    if( !f ) { std::cerr << "cannot read " << a[ "load" ] << std::endl; return 1; }

    if( d == 2 ) return run<OptimalTransportMap2d>( d, n, pos, w, a );
    if( d == 3 ) return run<OptimalTransportMap3d>( d, n, pos, w, a );
    std::cerr << "dimension " << d << ": geogram's OTM is 2D or 3D" << std::endl;
    return 1;
}
