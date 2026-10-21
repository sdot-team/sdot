#pragma once

// =====================================================================================
// THE GLUING OF A CELL COMPLEX ( `CellComplex.py` ), any dimension: the true face lattice of each cell, then a name per face that every
// cell having it gives alike.
//
// PER CELL. The cells of the kernel are SIMPLE polytopes ( each vertex the corner of exactly `D` cuts ): a degenerate cell -- a cube
// of a grid, `2^D` cells around each corner -- is represented with extra vertices at one place, zero-length edges, cuts that touch
// only a lower face. The lattice is read off the GEOMETRY instead, which is that of a convex polytope whatever its representation:
//
//   * the POINTS: the local vertices merged by their global number ( `gv`, the vertex gluing ) when at one place;
//   * the incidences: a point is on the cuts of the local vertices merged into it ( the planes, exact from the seeds, give the
//     normals );
//   * the true FACETS: the cuts whose points span `D - 1` dimensions; the true VERTICES: the points whose facets' normals span `D`
//     ( a point inside an edge or a facet is a corner of nothing );
//   * the FACES: the intersections of the vertex sets of the facets -- the children of a face are the sets `V( F ) n V( f )`, `f` a
//     facet not containing it, whose dimension is one less: `D - rank` of the normals of the facets containing them.
//
// GLOBALLY. A face is named by its points: their global numbers, sorted ( without a periodicity ). With a periodicity, a face may
// hold two copies of one vertex: each point then comes with the element of the group that relates it to the face's first point
// ( `E`, from the vertex gluing ), and a second name -- the generators of the facets containing the face, in the frame of the
// smallest seed among them ( exact whatever the frame, and the points the group fixes ) -- joins what the first splits. A face on
// a made-up wall ( `INFINITE` ) is none; a face with a made-up point ( unbounded ) is named by its finite points and its generators.
//
// Per true vertex: `D` of its facets with independent normals, the equations `CellGeometry` solves it from.
// =====================================================================================

#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

namespace sdot {
namespace glue {

using I64 = std::int64_t;

template<class F>
void par_for( const CpuQueue &queue, SI n, F &&f ) {
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), std::max<SI>( n / 64, 1 ) ) );
    if ( nt <= 1 ) {
        for ( SI i = 0; i < n; ++i ) f( i );
        return;
    }
    queue.run_threads( nt, [&]( int t ) {
        const SI b = n * t / nt, e = n * ( t + 1 ) / nt;
        for ( SI i = b; i < e; ++i ) f( i );
    } );
}

/// the rank of vectors ( rows of `D` ): Gram-Schmidt with PIVOTING -- at each step the vector with the largest residual, the others
/// orthogonalized against it -- and a threshold on that residual. Without the pivoting, two nearly collinear vectors taken first
/// ( a thin facet ) define their plane badly, and the residual of a third, on that plane, amplifies the rounding by their ratio
template<int D>
int rank_of( const std::vector<std::array<double,D>> &vs, double thr, std::vector<std::array<double,D>> *basis = nullptr ) {
    std::vector<std::array<double,D>> r = vs, b;
    std::vector<char> used( r.size(), 0 );
    while ( int( b.size() ) < D ) {
        int best = -1;
        double bn = thr;
        for ( int i = 0; i < int( r.size() ); ++i ) {
            if ( used[ i ] ) continue;
            double n = 0;
            for ( int d = 0; d < D; ++d ) n += r[ i ][ d ] * r[ i ][ d ];
            n = std::sqrt( n );
            if ( n > bn ) { bn = n; best = i; }
        }
        if ( best < 0 ) break;
        used[ best ] = 1;
        std::array<double,D> q = r[ best ];
        for ( int d = 0; d < D; ++d ) q[ d ] /= bn;
        for ( int i = 0; i < int( r.size() ); ++i ) {
            if ( used[ i ] ) continue;
            double s = 0;
            for ( int d = 0; d < D; ++d ) s += r[ i ][ d ] * q[ d ];
            for ( int d = 0; d < D; ++d ) r[ i ][ d ] -= s * q[ d ];
        }
        b.push_back( q );
    }
    if ( basis ) *basis = b;
    return int( b.size() );
}

/// THE LATTICE OF ONE CELL
template<int D>
struct CellLattice {
    // the points: a representative local vertex, the global number ( -1: made up ), the local vertices merged into it
    std::vector<SI>  rep;
    std::vector<I64> pgv;
    std::vector<int> fake;
    // the faces, all dimensions: dimension, points ( sorted ), facets containing them ( cut indices, sorted ), the cut of a facet
    std::vector<int> dim;
    std::vector<std::vector<int>> pts;
    std::vector<std::vector<SI>>  facets;
    std::vector<SI>  cut;
    std::vector<std::pair<int,int>> children;            // ( parent, child ), local face indices
    std::vector<std::array<SI,D>> eqs;                    // per face of dimension 0: the cuts of its equations
    bool on_infinite_ok = true;
};

template<int D>
CellLattice<D> lattice_of( SI cell, const auto &W, const auto &VC, const auto &gv, const auto &planes, const auto &cut_inf,
                           SI v0, SI v1, SI c0, SI c1, double tol, double same ) {
    CellLattice<D> L;
    // ---- the points
    std::vector<int> pt_of( v1 - v0, -1 );
    for ( SI v = v0; v < v1; ++v ) {
        const I64 g = I64( gv( v ) );
        int found = -1;
        if ( g >= 0 )
            for ( int p = 0; p < int( L.rep.size() ); ++p ) {
                if ( L.pgv[ p ] != g ) continue;
                double m = 0;
                for ( int d = 0; d < D; ++d ) m = std::max( m, std::fabs( double( W( v, d ) ) - double( W( L.rep[ p ], d ) ) ) );
                if ( m <= same ) { found = p; break; }
            }
        if ( found < 0 ) {
            found = int( L.rep.size() );
            L.rep.push_back( v );
            L.pgv.push_back( g );
            L.fake.push_back( g < 0 );
        }
        pt_of[ v - v0 ] = found;
    }
    const int np = int( L.rep.size() ), nc = int( c1 - c0 );
    auto P = [&]( int p, int d ) { return double( W( L.rep[ p ], d ) ); };

    // ---- the incidences: by the representation ( the cuts of each local vertex ) and by the geometry ( within `tol` of the plane )
    std::vector<std::vector<int>> inc( nc );                 // points of each cut, sorted
    std::vector<std::vector<int>> rep_inc( nc );
    for ( SI v = v0; v < v1; ++v )
        for ( int r = 0; r < D; ++r ) {
            const SI c = SI( VC( v, r ) ) - c0;
            if ( c >= 0 && c < nc ) rep_inc[ c ].push_back( pt_of[ v - v0 ] );
        }
    // ( by the representation only: a cut through an existing vertex makes a copy of it on the crossing edge, which the merge of the
    // copies joins -- and a test by distance is no test at a vertex where planes meet at angles of 1e-9: two points 1e-3 apart are
    // then both within 1e-12 of all of them )
    for ( int c = 0; c < nc; ++c ) {
        std::vector<int> &s = inc[ c ];
        s = rep_inc[ c ];
        std::sort( s.begin(), s.end() );
        s.erase( std::unique( s.begin(), s.end() ), s.end() );
    }

    auto normal = [&]( int c ) {
        std::array<double,D> n;
        for ( int d = 0; d < D; ++d ) n[ d ] = double( planes( c0 + c, d ) );
        return n;
    };
    // the true facets: the cuts whose points span `D - 1` dimensions -- told by the NORMALS, exact from the seeds and the same in every
    // cell: `D - rank` of the normals of all the cuts containing all their points. Not by the positions: at a vertex where planes meet
    // at angles of 1e-9, its copies in the cells around it differ by 1e-7, and a sliver of 1e-10 is a facet for one and not another
    std::vector<int> facet;                                  // ( local cut indices )
    for ( int c = 0; c < nc; ++c ) {
        if ( inc[ c ].size() < std::size_t( D ) ) continue;
        std::vector<std::array<double,D>> ns;
        for ( int e = 0; e < nc; ++e )
            if ( ! inc[ e ].empty() && std::includes( inc[ e ].begin(), inc[ e ].end(), inc[ c ].begin(), inc[ c ].end() ) )
                ns.push_back( normal( e ) );
        if ( rank_of<D>( ns, 1e-12 ) == 1 )
            facet.push_back( c );
    }
    // facets containing a set of points ( sorted )
    auto facets_of = [&]( const std::vector<int> &s ) {
        std::vector<int> res;
        for ( int c : facet )
            if ( std::includes( inc[ c ].begin(), inc[ c ].end(), s.begin(), s.end() ) )
                res.push_back( c );
        return res;
    };
    auto dim_of = [&]( const std::vector<int> &fs ) {
        std::vector<std::array<double,D>> ns;
        for ( int c : fs ) ns.push_back( normal( c ) );
        return D - rank_of<D>( ns, 1e-12 );
    };

    // ---- the faces, from the cell down
    std::vector<std::map<std::vector<int>,int>> seen( D + 1 );
    auto add = [&]( int k, const std::vector<int> &s, const std::vector<int> &fs, SI cut ) {
        auto it = seen[ k ].find( s );
        if ( it != seen[ k ].end() ) return it->second;
        const int id = int( L.dim.size() );
        seen[ k ][ s ] = id;
        L.dim.push_back( k );
        L.pts.push_back( s );
        std::vector<SI> f;
        for ( int c : fs ) f.push_back( c0 + c );
        L.facets.push_back( f );
        L.cut.push_back( cut );
        return id;
    };
    {
        std::vector<int> all;
        for ( int p = 0; p < np; ++p ) {
            // ( a point counts if it is a corner: its facets span `D` )
            if ( dim_of( facets_of( { p } ) ) == 0 ) all.push_back( p );
        }
        if ( all.empty() ) return L;
        add( D, all, {}, -1 );
    }
    for ( int k = D; k >= 1; --k ) {
        // ( the faces of dimension `k` found so far, in order )
        std::vector<int> cur;
        for ( int i = 0; i < int( L.dim.size() ); ++i ) if ( L.dim[ i ] == k ) cur.push_back( i );
        for ( int fi : cur ) {
            const std::vector<int> V = L.pts[ fi ];
            std::vector<SI> SF = L.facets[ fi ];
            for ( int c : facet ) {
                if ( std::binary_search( SF.begin(), SF.end(), c0 + c ) ) continue;
                std::vector<int> C;
                std::set_intersection( V.begin(), V.end(), inc[ c ].begin(), inc[ c ].end(), std::back_inserter( C ) );
                if ( C.empty() ) continue;
                // ( only the corners: a point inside an edge is no vertex of its faces )
                std::vector<int> Ck;
                for ( int p : C ) if ( std::binary_search( L.pts[ 0 ].begin(), L.pts[ 0 ].end(), p ) ) Ck.push_back( p );
                if ( Ck.empty() ) continue;
                const std::vector<int> fs = facets_of( Ck );
                if ( dim_of( fs ) != k - 1 ) continue;
                const int child = add( k - 1, Ck, fs, k - 1 == D - 1 ? SI( c0 + fs[ 0 ] ) : SI( -1 ) );
                L.children.push_back( { fi, child } );
            }
        }
    }
    // a facet of dimension `D - 1` takes the cut whose plane it lies on: the first of its facets is the only one ( a true facet is one cut )
    for ( int i = 0; i < int( L.dim.size() ); ++i )
        if ( L.dim[ i ] == D - 1 && ! L.facets[ i ].empty() )
            L.cut[ i ] = L.facets[ i ][ 0 ];
    // the equations of the vertices: `D` of their facets, independent
    for ( int i = 0; i < int( L.dim.size() ); ++i ) {
        if ( L.dim[ i ] != 0 ) continue;
        std::array<SI,D> e;
        e.fill( -1 );
        std::vector<std::array<double,D>> chosen;
        int q = 0;
        for ( SI c : L.facets[ i ] ) {
            std::vector<std::array<double,D>> t = chosen;
            t.push_back( normal( int( c - c0 ) ) );
            if ( rank_of<D>( t, 1e-9 ) > int( chosen.size() ) ) {
                chosen.push_back( normal( int( c - c0 ) ) );
                e[ q++ ] = c;
                if ( q == D ) break;
            }
        }
        L.eqs.push_back( e );
    }
    (void)cell;
    return L;
}

// ---- the names --------------------------------------------------------------------------------------

inline I64 round_key( double x ) { return I64( std::llround( x * 1048576.0 ) ); }

/// affine maps `[ D, D + 1 ]`, row-major
template<int D> using Aff = std::array<double,D*(D+1)>;

template<int D>
Aff<D> compose( const Aff<D> &a, const Aff<D> &b ) {
    Aff<D> r;
    for ( int i = 0; i < D; ++i ) {
        for ( int j = 0; j < D; ++j ) {
            double s = 0;
            for ( int k = 0; k < D; ++k ) s += a[ i * ( D + 1 ) + k ] * b[ k * ( D + 1 ) + j ];
            r[ i * ( D + 1 ) + j ] = s;
        }
        double s = a[ i * ( D + 1 ) + D ];
        for ( int k = 0; k < D; ++k ) s += a[ i * ( D + 1 ) + k ] * b[ k * ( D + 1 ) + D ];
        r[ i * ( D + 1 ) + D ] = s;
    }
    return r;
}

template<int D>
Aff<D> inverse( const Aff<D> &a ) {                         // ( orthogonal linear part )
    Aff<D> r;
    for ( int i = 0; i < D; ++i ) {
        for ( int j = 0; j < D; ++j ) r[ i * ( D + 1 ) + j ] = a[ j * ( D + 1 ) + i ];
        double s = 0;
        for ( int k = 0; k < D; ++k ) s -= a[ k * ( D + 1 ) + i ] * a[ k * ( D + 1 ) + D ];
        r[ i * ( D + 1 ) + D ] = s;
    }
    return r;
}

template<int D>
Aff<D> aff_at( const auto &T, SI i ) {
    Aff<D> r;
    for ( int a = 0; a < D; ++a ) for ( int b = 0; b <= D; ++b ) r[ a * ( D + 1 ) + b ] = double( T( i, a, b ) );
    return r;
}

/// THE GENERATOR NAME of a face: its own seed and the generators of its facets ( seeds with their images, walls with their planes ),
/// in the frame of the smallest seed among them -- the smallest over the choices of that seed's copies
template<int D>
std::vector<I64> generator_name( SI own, const std::vector<SI> &facets, const auto &cgen, const auto &cimg, const auto &G, const auto &walls ) {
    struct Gen { I64 seed; Aff<D> e; int wall; };
    std::vector<Gen> gens;
    Aff<D> id{};
    for ( int i = 0; i < D; ++i ) id[ i * ( D + 1 ) + i ] = 1;
    gens.push_back( { I64( own ), id, -1 } );
    for ( SI c : facets ) {
        const I64 j = I64( cgen( c ) );
        if ( j >= 0 ) gens.push_back( { j, aff_at<D>( G, SI( cimg( c ) ) ), -1 } );
        else gens.push_back( { -1, id, int( -1 - j ) } );
    }
    I64 low = -1;
    for ( const Gen &g : gens ) if ( g.seed >= 0 && ( low < 0 || g.seed < low ) ) low = g.seed;
    std::vector<I64> best;
    for ( const Gen &a : gens ) {
        if ( a.seed != low ) continue;
        const Aff<D> inv = inverse<D>( a.e );
        std::vector<std::vector<I64>> rows;
        for ( const Gen &g : gens ) {
            std::vector<I64> r;
            if ( g.seed >= 0 ) {
                r.push_back( g.seed );
                const Aff<D> m = compose<D>( inv, g.e );
                for ( double x : m ) r.push_back( round_key( x ) );
            } else {                                         // the wall `n . x <= o` in the frame `x = Ea y`: `( L^T n ) . y <= o - n . V`
                r.push_back( -1 );
                double o = double( walls( g.wall, D ) );
                for ( int k = 0; k < D; ++k ) o -= double( walls( g.wall, k ) ) * a.e[ k * ( D + 1 ) + D ];
                for ( int i = 0; i < D; ++i ) {
                    double s = 0;
                    for ( int k = 0; k < D; ++k ) s += a.e[ k * ( D + 1 ) + i ] * double( walls( g.wall, k ) );
                    r.push_back( round_key( s ) );
                }
                r.push_back( round_key( o ) );
                while ( r.size() < std::size_t( 1 + D * ( D + 1 ) ) ) r.push_back( 0 );
            }
            rows.push_back( r );
        }
        std::sort( rows.begin(), rows.end() );
        std::vector<I64> flat;
        for ( const auto &r : rows ) flat.insert( flat.end(), r.begin(), r.end() );
        if ( best.empty() || flat < best ) best = flat;
    }
    return best;
}

/// the union-find of the names
struct Union {
    std::vector<int> p;
    explicit Union( int n ) : p( n ) { std::iota( p.begin(), p.end(), 0 ); }
    int find( int a ) { while ( p[ a ] != a ) a = p[ a ] = p[ p[ a ] ]; return a; }
    void join( int a, int b ) { a = find( a ); b = find( b ); if ( a != b ) p[ std::max( a, b ) ] = std::min( a, b ); }
};

/// THE ENTRY POINT. Outputs ( capacities `cap_f` faces, `cap_p` points of faces, `cap_c` children, `cap_v` vertices ):
///   `f_cell`, `f_dim`, `f_gid`, `f_cut`, `f_fake` per local face, `f_off` / `f_pts` its points ( representative local vertices, the
///   made-up ones included ),
///   `c_par` / `c_chi` the local children, `v_eq` the equations of each local face of dimension 0 ( in the order of the faces ),
///   `stats`: the numbers of faces, points, children, vertices, then the number of global faces per dimension.
template<int D>
void run( const CpuQueue &queue, const auto &W, const auto &VC, const auto &gv, const auto &E, const auto &planes, const auto &cut_inf,
          const auto &cgen, const auto &cimg, const auto &cell_v, const auto &cell_c, const auto &G, const auto &walls, const auto &params,
          auto &&f_cell, auto &&f_dim, auto &&f_gid, auto &&f_cut, auto &&f_fake, auto &&f_off, auto &&f_pts, auto &&c_par, auto &&c_chi,
          auto &&v_eq, auto &&stats ) {
    const SI n = SI( cell_v.shape( 0 ) ) - 1;
    const double tol = double( params( 0 ) ), same = double( params( 1 ) );
    const bool periodic = double( params( 2 ) ) != 0;
    auto is_inf = [&]( SI c ) { return int( cut_inf( c ) ) != 0; };

    std::vector<CellLattice<D>> cells( n );
    par_for( queue, n, [&]( SI i ) {
        cells[ i ] = lattice_of<D>( i, W, VC, gv, planes, is_inf, SI( cell_v( i ) ), SI( cell_v( i + 1 ) ), SI( cell_c( i ) ), SI( cell_c( i + 1 ) ), tol, same );
    } );

    // a face on a made-up wall is none; one with a made-up point is unbounded
    struct Ref { SI cell; int face; };
    std::vector<std::vector<Ref>> by_dim( D + 1 );
    for ( SI i = 0; i < n; ++i ) {
        const auto &L = cells[ i ];
        for ( int f = 0; f < int( L.dim.size() ); ++f ) {
            bool on_inf = false;
            for ( SI c : L.facets[ f ] ) on_inf |= is_inf( c );
            if ( ! on_inf ) by_dim[ L.dim[ f ] ].push_back( { i, f } );
        }
    }

    // ---- the global names, per dimension
    std::vector<std::vector<I64>> gid( n );
    for ( SI i = 0; i < n; ++i ) gid[ i ].assign( cells[ i ].dim.size(), -1 );
    std::vector<I64> nb_glob( D + 1, 0 );
    for ( int k = 0; k <= D; ++k ) {
        const auto &R = by_dim[ k ];
        const int m = int( R.size() );
        if ( k == D ) {                                      // the cells are the seeds
            for ( const Ref &r : R ) gid[ r.cell ][ r.face ] = r.cell;
            nb_glob[ k ] = n;
            continue;
        }
        // the first name: the points ( global numbers, and with a periodicity the element relative to the first point )
        std::vector<std::vector<I64>> name( m );
        par_for( queue, m, [&]( SI q ) {
            const Ref &r = R[ q ];
            const auto &L = cells[ r.cell ];
            const auto &ps = L.pts[ r.face ];
            std::vector<std::vector<I64>> rows;
            bool unbounded = false;
            int ref = -1;
            for ( int p : ps ) {
                if ( L.fake[ p ] ) { unbounded = true; continue; }
                if ( ref < 0 || L.pgv[ p ] < L.pgv[ ref ] ) ref = p;
            }
            Aff<D> iref{};
            if ( periodic && ref >= 0 ) iref = inverse<D>( aff_at<D>( E, L.rep[ ref ] ) );
            for ( int p : ps ) {
                if ( L.fake[ p ] ) continue;
                std::vector<I64> row{ L.pgv[ p ] };
                if ( periodic ) {
                    const Aff<D> rel = compose<D>( iref, aff_at<D>( E, L.rep[ p ] ) );
                    for ( double x : rel ) row.push_back( round_key( x ) );
                }
                rows.push_back( row );
            }
            std::sort( rows.begin(), rows.end() );
            std::vector<I64> flat;
            for ( const auto &row : rows ) flat.insert( flat.end(), row.begin(), row.end() );
            if ( unbounded ) {                               // ( and its generators: the finite points do not name an unbounded face )
                flat.push_back( -2 );
                std::vector<I64> gens{ I64( r.cell ) };
                for ( SI c : L.facets[ r.face ] ) gens.push_back( I64( cgen( c ) ) );
                std::sort( gens.begin(), gens.end() );
                flat.insert( flat.end(), gens.begin(), gens.end() );
            }
            name[ q ] = flat;
        } );
        Union u( m );
        auto join_equal = [&]( std::vector<std::vector<I64>> &names ) {
            std::vector<int> order( m );
            std::iota( order.begin(), order.end(), 0 );
            std::sort( order.begin(), order.end(), [&]( int a, int b ) { return names[ a ] < names[ b ]; } );
            for ( int q = 1; q < m; ++q )
                if ( names[ order[ q ] ] == names[ order[ q - 1 ] ] )
                    u.join( order[ q ], order[ q - 1 ] );
        };
        join_equal( name );
        // the second name, with a periodicity: the generators -- what joins the copies the points split
        if ( periodic && k > 0 ) {
            std::vector<std::vector<I64>> gname( m );
            par_for( queue, m, [&]( SI q ) {
                const Ref &r = R[ q ];
                gname[ q ] = generator_name<D>( r.cell, cells[ r.cell ].facets[ r.face ], cgen, cimg, G, walls );
            } );
            join_equal( gname );
        }
        std::vector<I64> label( m, -1 );
        I64 next = 0;
        std::vector<I64> of_root( m, -1 );
        for ( int q = 0; q < m; ++q ) {
            const int root = u.find( q );
            if ( of_root[ root ] < 0 ) of_root[ root ] = next++;
            gid[ R[ q ].cell ][ R[ q ].face ] = of_root[ root ];
        }
        nb_glob[ k ] = next;
    }

    // ---- the outputs
    const SI cap_f = SI( f_cell.shape( 0 ) ), cap_p = SI( f_pts.shape( 0 ) ), cap_c = SI( c_par.shape( 0 ) ), cap_v = SI( v_eq.shape( 0 ) );
    SI nf = 0, np = 0, nch = 0, nv = 0;
    for ( SI i = 0; i < n; ++i ) {
        const auto &L = cells[ i ];
        const SI base = nf;
        for ( int f = 0; f < int( L.dim.size() ); ++f ) {
            if ( nf < cap_f ) {
                f_cell( nf ) = int( i );
                f_dim( nf ) = L.dim[ f ];
                f_gid( nf ) = int( gid[ i ][ f ] );
                f_cut( nf ) = int( L.cut[ f ] );
                bool fk = false;
                for ( int p : L.pts[ f ] ) fk |= bool( L.fake[ p ] );
                f_fake( nf ) = int( fk );
                f_off( nf ) = int( np );
            }
            for ( int p : L.pts[ f ] ) {                    // ( the made-up points too: their global number is -1 )
                if ( np < cap_p ) f_pts( np ) = int( L.rep[ p ] );
                ++np;
            }
            ++nf;
        }
        for ( const auto &pc : L.children ) {
            if ( nch < cap_c ) { c_par( nch ) = int( base + pc.first ); c_chi( nch ) = int( base + pc.second ); }
            ++nch;
        }
        for ( const auto &e : L.eqs ) {
            if ( nv < cap_v ) for ( int d = 0; d < D; ++d ) v_eq( nv, d ) = int( e[ d ] );
            ++nv;
        }
    }
    if ( nf < cap_f ) f_off( nf ) = int( np );
    stats( 0 ) = nf; stats( 1 ) = np; stats( 2 ) = nch; stats( 3 ) = nv;
    for ( int k = 0; k <= D; ++k ) stats( 4 + k ) = nb_glob[ k ];
}

} // namespace glue
} // namespace sdot
