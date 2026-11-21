// The weighted semi-coarsened tent blurs of the grid ( sdot/grid/GridNewton2D.h `TensorPrecond`, grid_stencil/newton.cpp `tensor` )
// for the hexagonal lattice.
//
// The lattice in offset rows is HALF of a rectangular grid G ( a checkerboard ): seed ( r, c ) at G ( r, i = 2 c + ( r & 1 ) ), G of
// spacing 1/2 along x ( axis b, index i ) and sqrt( 3 ) / 2 along y ( axis a, index r ). The preconditioner is
//     z = r / diag + E^T ( sum over the subspaces ( la, lb ) != ( 0, 0 ) of P D R ) E r,
// E the injection of the lattice in G ( zero on the other half ), P the separable tents of widths 2^la ( rows ) and 2^lb ( columns )
// ( vertex-centred: f[ 2 A ] = c[ A ], f[ 2 A + 1 ] = ( c[ A ] + c[ A + 1 ] ) / 2, any size; the last point of an even line takes
// c[ A ] whole, else the constants are not reproduced and CG its doubled on even sizes ), R = P^T, and D per coarse point
//     D = balance( Ea, Eb ) / ( Ea + Eb ),  Ea = c R( ka ) / 4^la,  Eb = c R( kb ) / 4^lb,  balance = ( min / max )^pw,
// ka = sum_faces w dr^2, kb = sum_faces w di^2 the conductances of each cell along the two axes ( G index units ): a subspace whose
// tents have the shape of the cells carries the weight, the others little.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace hexgrid {

template<class T_>
struct HexTensorT {
    using T = T_;                                  // the storage and arithmetic of the blurs ( float halves the traffic ); r, z stay double
    int R = 0, NB = 0, La = 0, Lb = 0;             // G: R rows ( axis a ) x NB columns ( axis b )
    T c = T( 2 ) / 3, pw = T( 0.5 );
    int band = 1 << 20;
    std::vector<int> na, nb;                       // sizes per level
    std::vector<std::vector<T>> g, Di;             // ( La + 1 ) x ( Lb + 1 ) grids
    std::vector<T> id0;
    int gi( int la, int lb ) const { return la * ( Lb + 1 ) + lb; }
    int drop = 0;                                  // ( experiment, HEX_DROP: bit 1 no ( 0, 1 ), bit 2 no ( 1, 0 ), bit 4 no ( la, 0 ) at all: the
                                                   //   base of the tensor is G( 0, 1 ), spacing 1 x sqrt3/2, nearly isotropic; bands counted from it )
    T fine_w = 1;                                  // ( experiment, HEX_FINEW: the weight of ( 0, 1 ) and ( 1, 0 ) )
    bool kept( int la, int lb ) const { return ( la || lb ) && ! ( ( drop & 4 ) && lb == 0 ) && std::abs( la - ( drop & 4 ? lb - 1 : lb ) ) <= band && ! ( ( drop & 1 ) && la == 0 && lb == 1 ) && ! ( ( drop & 2 ) && la == 1 && lb == 0 ); }

    static int half( int n ) { return ( n + 1 ) / 2; }
    // CELL-CENTRED tents, the grid's ( sdot/grid/GridNewton2D.h `Tents` ): coarse B = fine 2 B, 2 B + 1, restriction [ 1/4 3/4 3/4 1/4 ]
    // ( 1 at the ends of the line ), prolongation its transpose; an odd line has a virtual zero at its end
    bool cell = false;
    static T cw( int B, int m, int k ) { // weight of fine 2 B + k ( k = -1 .. 2 ) in coarse B
        if( m == 1 ) return k == 0 || k == 1 ? T( 1 ) : T( 0 );
        if( k == -1 ) return B > 0 ? T( 0.25 ) : T( 0 );
        if( k == 2 ) return B < m - 1 ? T( 0.25 ) : T( 0 );
        if( k == 0 ) return B == 0 ? T( 1 ) : T( 0.75 );
        return B == m - 1 ? T( 1 ) : T( 0.75 );
    }
    static void restr_c( const T *f, int n, int sf, T *c, int sc ) {
        const int m = half( n );
        for( int B = 0; B < m; ++B ) {
            T s = 0;
            for( int k = -1; k <= 2; ++k ) { const int j = 2 * B + k; if( j >= 0 && j < n ) s += cw( B, m, k ) * f[ j * sf ]; }
            c[ B * sc ] = s;
        }
    }
    static void prol_add_c( const T *c, int m, int sc, T *f, int n, int sf ) {
        for( int B = 0; B < m; ++B )
            for( int k = -1; k <= 2; ++k ) { const int j = 2 * B + k; if( j >= 0 && j < n ) f[ j * sf ] += cw( B, m, k ) * c[ B * sc ]; }
    }
    // 1D tents on a strided line: c ( m ) = R f ( n ), f ( n ) += P c ( m )
    static void restr( const T *f, int n, int sf, T *c, int sc ) {
        const int m = half( n );
        if( sf == 1 && sc == 1 && n >= 3 ) {       // ( contiguous: branch-free interior, the two ends apart )
            c[ 0 ] = f[ 0 ] + ( m > 1 ? T( 0.5 ) : T( 1 ) ) * f[ 1 ];
            const int Ae = ( n % 2 ) ? m - 1 : m - 1; // interior: 1 <= A < m - 1 ( both neighbours, weight 1 / 2 )
            for( int A = 1; A < Ae; ++A ) c[ A ] = f[ 2 * A ] + T( 0.5 ) * ( f[ 2 * A - 1 ] + f[ 2 * A + 1 ] );
            const int A = m - 1;
            T sl = f[ 2 * A ] + T( 0.5 ) * f[ 2 * A - 1 ];
            if( 2 * A + 1 < n ) sl += f[ 2 * A + 1 ];   // ( n even: the last fine point has c[ A ] alone )
            c[ A ] = sl;
            return;
        }
        for( int A = 0; A < m; ++A ) {
            T s = f[ 2 * A * sf ];
            if( 2 * A + 1 < n ) s += ( A + 1 < m ? T( 0.5 ) : T( 1 ) ) * f[ ( 2 * A + 1 ) * sf ];   // ( n even: the last fine point has c[ A ] alone )
            if( 2 * A - 1 >= 0 ) s += T( 0.5 ) * f[ ( 2 * A - 1 ) * sf ];
            c[ A * sc ] = s;
        }
    }
    static void prol_add( const T *c, int m, int sc, T *f, int n, int sf ) {
        if( sf == 1 && sc == 1 && m >= 2 ) {
            // ( one loop writing both, interleaved: NEON st2 / ld2 )
            for( int A = 0; A < m - 1; ++A ) { f[ 2 * A ] += c[ A ]; f[ 2 * A + 1 ] += T( 0.5 ) * ( c[ A ] + c[ A + 1 ] ); }
            f[ 2 * ( m - 1 ) ] += c[ m - 1 ];
            if( 2 * ( m - 1 ) + 1 < n ) f[ 2 * ( m - 1 ) + 1 ] += c[ m - 1 ];
            return;
        }
        for( int A = 0; A < m; ++A ) f[ 2 * A * sf ] += c[ A * sc ];
        for( int A = 0; 2 * A + 1 < n; ++A ) f[ ( 2 * A + 1 ) * sf ] += A + 1 < m ? T( 0.5 ) * ( c[ A * sc ] + c[ ( A + 1 ) * sc ] ) : c[ A * sc ]; // ( constants reproduced )
    }
    // grid ( la, lb ) -> ( la + 1, lb ) / ( la, lb + 1 ), and back
    // ( along a: whole rows at once, contiguous, vectorized )
    void restrict_a( int la, int lb ) {
        if( cell ) { const int A = na[ la ], Bn = nb[ lb ]; T *f = g[ gi( la, lb ) ].data(), *cg = g[ gi( la + 1, lb ) ].data(); for( int b = 0; b < Bn; ++b ) restr_c( f + b, A, Bn, cg + b, Bn ); return; }
        const int n = na[ la ], m = na[ la + 1 ], Bn = nb[ lb ];
        const T *f = g[ gi( la, lb ) ].data(); T *cg = g[ gi( la + 1, lb ) ].data();
        for( int A = 0; A < m; ++A ) {
            T *c = cg + size_t( A ) * Bn;
            const T *f0 = f + size_t( 2 * A ) * Bn;
            const T w1 = 2 * A + 1 < n ? ( A + 1 < m ? T( 0.5 ) : T( 1 ) ) : T( 0 ), wm = A ? T( 0.5 ) : T( 0 );
            const T *f1 = 2 * A + 1 < n ? f0 + Bn : f0, *fm = A ? f0 - Bn : f0;
            for( int b = 0; b < Bn; ++b ) c[ b ] = f0[ b ] + w1 * f1[ b ] + wm * fm[ b ];
        }
    }
    void restrict_b( int la, int lb ) { const int A = na[ la ], Bn = nb[ lb ], m = nb[ lb + 1 ]; T *f = g[ gi( la, lb ) ].data(), *cg = g[ gi( la, lb + 1 ) ].data(); for( int a = 0; a < A; ++a ) { if( cell ) restr_c( f + a * Bn, Bn, 1, cg + a * m, 1 ); else restr( f + a * Bn, Bn, 1, cg + a * m, 1 ); } }
    void prolong_a( int la, int lb ) {
        if( cell ) { const int A = na[ la ], Bn = nb[ lb ]; T *f = g[ gi( la, lb ) ].data(), *cg = g[ gi( la + 1, lb ) ].data(); for( int b = 0; b < Bn; ++b ) prol_add_c( cg + b, na[ la + 1 ], Bn, f + b, A, Bn ); return; }
        const int n = na[ la ], m = na[ la + 1 ], Bn = nb[ lb ];
        T *f = g[ gi( la, lb ) ].data(); const T *cg = g[ gi( la + 1, lb ) ].data();
        for( int A = 0; A < m; ++A ) {
            const T *c0 = cg + size_t( A ) * Bn, *c1 = A + 1 < m ? c0 + Bn : c0;
            T *f0 = f + size_t( 2 * A ) * Bn;
            for( int b = 0; b < Bn; ++b ) f0[ b ] += c0[ b ];
            if( 2 * A + 1 < n ) {
                T *f1 = f0 + Bn;
                if( A + 1 < m ) for( int b = 0; b < Bn; ++b ) f1[ b ] += T( 0.5 ) * ( c0[ b ] + c1[ b ] );
                else for( int b = 0; b < Bn; ++b ) f1[ b ] += c0[ b ];
            }
        }
    }
    // f = D f + P c along b ( row by row: the row scaled, then the coarse row added )
    void prolong_b_scaled( int la, int lb ) {
        const int A = na[ la ], Bn = nb[ lb ], m = nb[ lb + 1 ];
        T *f = g[ gi( la, lb ) ].data(); const T *cg = g[ gi( la, lb + 1 ) ].data();
        const bool has = ! Di[ gi( la, lb ) ].empty();
        const T *d = has ? Di[ gi( la, lb ) ].data() : nullptr;
        for( int a = 0; a < A; ++a ) {
            T *fr = f + size_t( a ) * Bn;
            if( has ) { const T *dr = d + size_t( a ) * Bn; for( int b = 0; b < Bn; ++b ) fr[ b ] *= dr[ b ]; }
            else std::fill( fr, fr + Bn, T( 0 ) );
            prol_add( cg + size_t( a ) * m, m, 1, fr, Bn, 1 );
        }
    }
    void prolong_b( int la, int lb ) { const int A = na[ la ], Bn = nb[ lb ], m = nb[ lb + 1 ]; T *f = g[ gi( la, lb ) ].data(), *cg = g[ gi( la, lb + 1 ) ].data(); for( int a = 0; a < A; ++a ) { if( cell ) prol_add_c( cg + a * m, m, 1, f + a * Bn, Bn, 1 ); else prol_add( cg + a * m, m, 1, f + a * Bn, Bn, 1 ); } }

    // the whole tree from g( 0, 0 ): first along a ( lb = 0 ), then along b from each
    // ( mode 3, `shift`: the odd rows are at x = c + 1/2; before the first transfer along a they are brought to x = c,
    // v'[ c ] = ( v[ c - 1 ] + v[ c ] ) / 2, and the prolongation takes the transpose: the coarse rows are aligned )
    bool shift = false;
    std::vector<T> tmp0;
    void shift_odd( const T *f, T *o ) const {          // o = S f ( all rows; even rows copied )
        for( int r = 0; r < R; ++r ) {
            const T *fr = f + size_t( r ) * NB; T *orow = o + size_t( r ) * NB;
            if( !( r & 1 ) ) { std::copy( fr, fr + NB, orow ); continue; }
            orow[ 0 ] = fr[ 0 ];
            for( int c = 1; c < NB; ++c ) orow[ c ] = T( 0.5 ) * ( fr[ c - 1 ] + fr[ c ] );
        }
    }
    void shift_odd_t_add( const T *a, T *f ) const {    // f += S^T a
        for( int r = 0; r < R; ++r ) {
            const T *ar = a + size_t( r ) * NB; T *fr = f + size_t( r ) * NB;
            if( !( r & 1 ) ) { for( int c = 0; c < NB; ++c ) fr[ c ] += ar[ c ]; continue; }
            fr[ 0 ] += ar[ 0 ];
            for( int c = 1; c < NB; ++c ) { fr[ c - 1 ] += T( 0.5 ) * ar[ c ]; fr[ c ] += T( 0.5 ) * ar[ c ]; }
        }
    }
    void restrict_tree() {
        if( shift && mode == 3 ) {
            tmp0.resize( g[ 0 ].size() );
            shift_odd( g[ 0 ].data(), tmp0.data() );
            std::swap( g[ 0 ], tmp0 );
            restrict_a( 0, 0 );
            std::swap( g[ 0 ], tmp0 );
            for( int la = 1; la < La; ++la ) restrict_a( la, 0 );
            for( int la = 0; la <= La; ++la ) for( int lb = 0; lb < Lb; ++lb ) restrict_b( la, lb );
            return;
        }
        for( int la = 0; la < La; ++la ) restrict_a( la, 0 );
        for( int la = 0; la <= La; ++la ) for( int lb = 0; lb < Lb; ++lb ) restrict_b( la, lb );
    }
    void prolong_tree() {
        if( shift && mode == 3 ) {
            for( int la = 0; la <= La; ++la ) for( int lb = Lb - 1; lb >= 0; --lb ) prolong_b( la, lb );
            for( int la = La - 1; la >= 1; --la ) prolong_a( la, 0 );
            tmp0.assign( g[ 0 ].size(), T( 0 ) );
            std::swap( g[ 0 ], tmp0 );                     // ( the a-part into a zero grid, then S^T into g( 0, 0 ) )
            prolong_a( 0, 0 );
            std::swap( g[ 0 ], tmp0 );
            shift_odd_t_add( tmp0.data(), g[ 0 ].data() );
            return;
        }
        for( int la = 0; la <= La; ++la ) for( int lb = Lb - 1; lb >= 0; --lb ) prolong_b( la, lb );
        for( int la = La - 1; la >= 0; --la ) prolong_a( la, 0 );
    }

    // the embedding: 0 the checkerboard G ( i = 2 c + ( r & 1 ) ), 1 AXIAL ( i = c - floor( r / 2 ) + qoff: the lattice is a square
    // index grid with one diagonal, the box a parallelogram in it, the holes in two corners only )
    int mode = 0, qoff = 0;
    // ( mode 3: the OFFSET ROWS themselves, R x Wc, no hole: spacing 1 x sqrt3/2, the half shift of the odd rows smoothed by the tents )
    // ( mode 5: a FRAME of two hex axes: ( alpha, beta ) = fm ( q, r ) in axial coordinates, integers, shifted to start at 0 )
    int fm[ 4 ] = { 1, 0, 0, 1 }, amin = 0, bmin = 0, GR = 0;
    static int axq( int r, int c ) { return c - ( r - ( r & 1 ) ) / 2; }
    int gidx( int r, int c, int ) const {
        if( mode == 5 ) { const int q = axq( r, c ); return ( fm[ 0 ] * q + fm[ 1 ] * r - amin ) * NB + fm[ 2 ] * q + fm[ 3 ] * r - bmin; }
        return mode >= 2 ? r * NB + c : mode ? r * NB + c - ( r - ( r & 1 ) ) / 2 + qoff : r * NB + 2 * c + ( r & 1 );
    }
    void alloc( int R_, int Wc ) {
        R = R_;
        qoff = ( R - 1 - ( ( R - 1 ) & 1 ) ) / 2;
        NB = mode >= 2 ? Wc : mode ? Wc + qoff : 2 * Wc + 1;   // ( mode 2: a SQUARE lattice, G is the grid itself: to check this code against TensorPrecond )
        GR = R;
        if( mode == 5 ) {
            int a0 = 1 << 30, a1 = -a0, b0 = a0, b1 = -a0;
            for( int r = 0; r < R; ++r ) for( int c : { 0, Wc - 1 } ) { const int q = axq( r, c ), a = fm[ 0 ] * q + fm[ 1 ] * r, b = fm[ 2 ] * q + fm[ 3 ] * r; a0 = std::min( a0, a ); a1 = std::max( a1, a ); b0 = std::min( b0, b ); b1 = std::max( b1, b ); }
            amin = a0; bmin = b0; GR = a1 - a0 + 1; NB = b1 - b0 + 1;
        }
        na = { GR }; while( na.back() > 2 ) na.push_back( half( na.back() ) ); La = int( na.size() ) - 1;
        nb = { NB }; while( nb.back() > 2 ) nb.push_back( half( nb.back() ) ); Lb = int( nb.size() ) - 1;
        g.assign( ( La + 1 ) * ( Lb + 1 ), {} ); Di.assign( g.size(), {} );
        for( int la = 0; la <= La; ++la ) for( int lb = 0; lb <= Lb; ++lb ) g[ gi( la, lb ) ].assign( size_t( na[ la ] ) * nb[ lb ], 0 );
    }


    // E as an INTERPOLATION ( `fill` ): the points of G that are not seeds take the mean of their neighbours along the row ( one at
    // the ends ), instead of 0; E^T sends each half back. ( experiment: zero injection makes the tents average with holes )
    bool fill = false;
    // seed c of a row of parity p sits at i = 2 c + p; a hole ( parity 1 - p ) takes the mean of its valid seed neighbours i -+ 1
    int hole_nb( int i, int p ) const { // number of valid seed neighbours of hole i
        int k = 0;
        for( int j : { i - 1, i + 1 } ) if( j >= 0 && j < NB && ( ( j - p ) & 1 ) == 0 && ( j - p ) / 2 < Wc_ ) ++k;
        return k;
    }
    void fill_g0() {
        T *f = g[ 0 ].data();
        for( int r = 0; r < R; ++r ) {
            T *row = f + size_t( r ) * NB;
            const int p = r & 1;
            for( int i = 1 - p; i < NB; i += 2 ) {
                T s = 0; int k = 0;
                for( int j : { i - 1, i + 1 } ) if( j >= 0 && j < NB && ( j - p ) / 2 < Wc_ ) { s += row[ j ]; ++k; }
                row[ i ] = k ? s / T( k ) : T( 0 );
            }
        }
    }
    void fill_g0_transpose_into( double *z, int Wc ) { // z[ c ] += the shares of the holes beside seed c ( g( 0, 0 ) after the prolongation )
        const T *f = g[ 0 ].data();
        for( int r = 0; r < R; ++r ) {
            const T *row = f + size_t( r ) * NB;
            const int p = r & 1;
            for( int c0 = 0; c0 < Wc; ++c0 ) {
                const int i = 2 * c0 + p;
                T s = 0;
                for( int h : { i - 1, i + 1 } ) if( h >= 0 && h < NB ) s += row[ h ] / T( hole_nb( h, p ) );
                z[ r * Wc + c0 ] += s;
            }
        }
    }
    int Wc_ = 0;

    /// diag: the lattice diagonal; ka, kb: per lattice cell ( R x Wc ) the conductances along rows / columns
    void build( int R_, int Wc, const std::vector<double> &diag, const std::vector<double> &ka, const std::vector<double> &kb ) {
        if( R != R_ || g.empty() ) alloc( R_, Wc );
        id0.resize( diag.size() );
        for( size_t i = 0; i < diag.size(); ++i ) id0[ i ] = diag[ i ] > 0 ? 1 / diag[ i ] : 0;
        for( int pass = 0; pass < 2; ++pass ) {
            std::fill( g[ 0 ].begin(), g[ 0 ].end(), T( 0 ) );
            for( int r = 0; r < R; ++r ) for( int c0 = 0; c0 < Wc; ++c0 ) g[ 0 ][ gidx( r, c0, NB ) ] = ( pass ? kb : ka )[ r * Wc + c0 ];
            Wc_ = Wc;
            if( fill ) fill_g0();
            restrict_tree();
            for( int la = 0; la <= La; ++la )
                for( int lb = 0; lb <= Lb; ++lb ) {
                    if( ! kept( la, lb ) ) continue;
                    std::vector<T> &D = Di[ gi( la, lb ) ];
                    const std::vector<T> &v = g[ gi( la, lb ) ];
                    D.resize( v.size() );
                    const T sa = c / T( 1 << la ) / T( 1 << la ), sb = c / T( 1 << lb ) / T( 1 << lb );
                    for( size_t q = 0; q < v.size(); ++q ) {
                        if( ! pass ) { D[ q ] = sa * v[ q ]; continue; }
                        const T ea = D[ q ], eb = sb * v[ q ], d = ea + eb, lo = std::min( ea, eb ), hi = std::max( ea, eb );
                        const T bal = hi > 0 ? ( pw == T( 0.5 ) ? std::sqrt( lo / hi ) : std::pow( lo / hi, pw ) ) : 0;
                        D[ q ] = d > 0 ? bal / d : 0;
                        if( la + lb == 1 ) D[ q ] *= fine_w;
                    }
                }
        }
    }

    /// the same without G( 0, 0 ) ( 2 n values, half zeros ): ( 0, 1 ) and ( 1, 0 ) straight from the lattice and back. r, z: rows of Wc
    /// values at stride rs / zs. z = r / diag + the blurs ( `mean` += sum of z if given )
    double tt[ 5 ] = {};                           // ( timing: lattice -> ( 0, 1 ) / ( 1, 0 ), restrictions, scaling, prolongations, back )
    void apply_fast( int Wc, const double *r, int rs, double *z, int zs ) {
        auto c0_ = std::chrono::steady_clock::now();
        auto lap = [&]( int k ) { auto t = std::chrono::steady_clock::now(); tt[ k ] += std::chrono::duration<double>( t - c0_ ).count(); c0_ = t; };
        const int m1 = nb[ 1 ];                    // = Wc + 1
        T *g01 = g[ gi( 0, 1 ) ].data(), *g10 = g[ gi( 1, 0 ) ].data();
        // ( 0, 1 ): even rows c[ A ] = x[ A ], odd rows c[ A ] = ( x[ A - 1 ] + x[ A ] ) / 2
        for( int rr = 0; rr < R; ++rr ) {
            const double *x = r + size_t( rr ) * rs; T *c = g01 + size_t( rr ) * m1;
            if( rr & 1 ) { c[ 0 ] = T( 0.5 ) * x[ 0 ]; for( int A = 1; A < Wc; ++A ) c[ A ] = T( 0.5 ) * ( x[ A - 1 ] + x[ A ] ); c[ Wc ] = T( 0.5 ) * x[ Wc - 1 ]; }
            else { for( int A = 0; A < Wc; ++A ) c[ A ] = x[ A ]; c[ Wc ] = 0; }
        }
        // ( 1, 0 ): row A = G row 2 A ( even i ) + w G rows 2 A +- 1 ( odd i )
        const int ma = na[ 1 ];
        for( int A = 0; A < ma; ++A ) {
            T *c = g10 + size_t( A ) * NB;
            const double *x0 = r + size_t( 2 * A ) * rs;
            const T w1 = 2 * A + 1 < R ? ( A + 1 < ma ? T( 0.5 ) : T( 1 ) ) : T( 0 ), wm = A ? T( 0.5 ) : T( 0 );
            const double *x1 = 2 * A + 1 < R ? x0 + rs : x0, *xm = A ? x0 - rs : x0;
            for( int cc = 0; cc < Wc; ++cc ) { c[ 2 * cc ] = x0[ cc ]; c[ 2 * cc + 1 ] = w1 * x1[ cc ] + wm * xm[ cc ]; }
            c[ 2 * Wc ] = 0;
        }
        lap( 0 );
        // the tree from there
        for( int la = 1; la < La; ++la ) restrict_a( la, 0 );
        for( int la = 0; la <= La; ++la ) for( int lb = la ? 0 : 1; lb < Lb; ++lb ) restrict_b( la, lb );
        lap( 1 );
        lap( 2 );
        // the scaling fused with the prolongations: each grid is scaled right before its children are added to it ( the top of each
        // b chain alone )
        auto scale = [&]( int la, int lb ) {
            std::vector<T> &v = g[ gi( la, lb ) ];
            if( Di[ gi( la, lb ) ].empty() ) { std::fill( v.begin(), v.end(), T( 0 ) ); return; }
            const T *d = Di[ gi( la, lb ) ].data(); T *vv = v.data(); const size_t nv = v.size();
            for( size_t k = 0; k < nv; ++k ) vv[ k ] *= d[ k ];
        };
        for( int la = 0; la <= La; ++la ) {
            scale( la, Lb );
            for( int lb = Lb - 1; lb >= ( la ? 0 : 1 ); --lb ) prolong_b_scaled( la, lb );
        }
        for( int la = La - 1; la >= 1; --la ) prolong_a( la, 0 );
        lap( 3 );
        // back to the lattice: the transposes of the two first transfers, plus r / diag
        for( int rr = 0; rr < R; ++rr ) {
            const double *x = r + size_t( rr ) * rs; const T *id = id0.data() + size_t( rr ) * Wc, *c = g01 + size_t( rr ) * m1;
            double *zr = z + size_t( rr ) * zs;
            if( rr & 1 ) for( int cc = 0; cc < Wc; ++cc ) zr[ cc ] = id[ cc ] * x[ cc ] + T( 0.5 ) * ( c[ cc ] + c[ cc + 1 ] );
            else for( int cc = 0; cc < Wc; ++cc ) zr[ cc ] = id[ cc ] * x[ cc ] + c[ cc ];
        }
        for( int A = 0; A < ma; ++A ) {
            const T *c0 = g10 + size_t( A ) * NB;
            double *z0 = z + size_t( 2 * A ) * zs;
            for( int cc = 0; cc < Wc; ++cc ) z0[ cc ] += c0[ 2 * cc ];
            if( 2 * A + 1 < R ) {
                const T *c1 = A + 1 < ma ? c0 + NB : c0;
                const T h = A + 1 < ma ? T( 0.5 ) : T( 1 );
                double *z1 = z0 + zs;
                for( int cc = 0; cc < Wc; ++cc ) z1[ cc ] += h * ( c0[ 2 * cc + 1 ] + ( A + 1 < ma ? c1[ 2 * cc + 1 ] : T( 0 ) ) );
            }
        }
        lap( 4 );
    }

    void apply( int Wc, const double *r, double *z ) {
        std::fill( g[ 0 ].begin(), g[ 0 ].end(), T( 0 ) );
        for( int rr = 0; rr < R; ++rr ) for( int c0 = 0; c0 < Wc; ++c0 ) g[ 0 ][ gidx( rr, c0, NB ) ] = r[ rr * Wc + c0 ];
        Wc_ = Wc;
        if( fill ) fill_g0();
        restrict_tree();
        for( size_t q = 1; q < g.size(); ++q ) {
            std::vector<T> &v = g[ q ];
            if( Di[ q ].empty() ) { std::fill( v.begin(), v.end(), T( 0 ) ); continue; }
            for( size_t k = 0; k < v.size(); ++k ) v[ k ] *= Di[ q ][ k ];
        }
        std::fill( g[ 0 ].begin(), g[ 0 ].end(), T( 0 ) );
        prolong_tree();
        for( int rr = 0; rr < R; ++rr ) for( int c0 = 0; c0 < Wc; ++c0 ) { const int i = rr * Wc + c0; z[ i ] = id0[ i ] * r[ i ] + g[ 0 ][ gidx( rr, c0, NB ) ]; }
        if( fill ) fill_g0_transpose_into( z, Wc );
    }
};

using HexTensor = HexTensorT<double>;

} // namespace hexgrid
