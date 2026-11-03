// sdot's own 2D cell engine ( Local2 + run<true>, i.e. the asimd register kernel ) fed by the
// same ring-R stencil, nearest first. Same outputs as ClipKernel ( hess per stencil offset ).
#pragma once
#include <sdot/cell/Engine.h>
#include <vector>
#include <cmath>

template<int R, class T = double>
struct SdotKernel {
    static constexpr int nb_offsets = ( 2 * R + 1 ) * ( 2 * R + 1 ) - 1, cap = 64;
    int ox[ nb_offsets ], oy[ nb_offsets ];

    struct Provider {
        const SdotKernel *k;
        const T *pi;
        int P, o = 0;
        template<class State> bool next( const State &, sdot::NothingLocal &, sdot::Plane<T,2> &p ) {
            if ( o >= nb_offsets ) return false;
            const int a = k->ox[ o ], b = k->oy[ o ];
            p.dir[ 0 ] = a; p.dir[ 1 ] = b;
            p.off = ( T( a * a + b * b ) + pi[ 0 ] - pi[ a * P + b ] ) / 2;
            p.id = o++;
            return true;
        }
    };
    struct Axes { T v[ 2 ][ 2 ]; T operator()( int i, int j ) const { return v[ i ][ j ]; } };

    SdotKernel() {
        int n = 0;
        for( int a = -R; a <= R; ++a ) for( int b = -R; b <= R; ++b ) if( a || b ) { ox[ n ] = a; oy[ n ] = b; ++n; }
        for( int i = 0; i < n; ++i ) for( int j = i + 1; j < n; ++j )
            if( ox[ j ] * ox[ j ] + oy[ j ] * oy[ j ] < ox[ i ] * ox[ i ] + oy[ i ] * oy[ i ] ) { std::swap( ox[ i ], ox[ j ] ); std::swap( oy[ i ], oy[ j ] ); }
    }

    void run( int N, const T *psi_pad, T *area, T *hess ) const {
        const int P = N + 2 * R;
        std::vector<std::int32_t> buf( sdot::Local2<T>::words_for( cap ) + 64 );
        for( int a = 0; a < N; ++a ) {
            for( int b = 0; b < N; ++b ) {
                sdot::Carver cv{ buf.data(), sdot::SI( buf.size() ) };
                sdot::Local2<T> c;
                c.attach( cv, cap );
                T origin[ 2 ] = { -a - T( 0.5 ), -b - T( 0.5 ) };
                Axes ax{ { { T( N ), 0 }, { 0, T( N ) } } };
                c.init_hypercube( origin, ax, -1 );
                Provider f{ this, psi_pad + ( a + R ) * P + ( b + R ), P };
                sdot::run<true>( c, f );
                area[ a * N + b ] = c.template measure<T>();
                T *hs = hess + ( a * N + b ) * nb_offsets;
                for( int o = 0; o < nb_offsets; ++o ) hs[ o ] = 0;
                c.template for_each_facet<T>( [ & ]( int q, T len ) {
                    int o = c.cid[ q ];
                    if( o >= 0 ) hs[ o ] += len / ( 2 * std::hypot( T( ox[ o ] ), T( oy[ o ] ) ) );
                } );
            }
        }
    }
};
