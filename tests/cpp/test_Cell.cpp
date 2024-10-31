#include "catch_main.h"
#include <sdot/Cell.h>

using namespace sdot;
struct Arch {};

void test_cut( auto &cell, const auto &dir, const auto &off, auto td, Vec<Vec<double>> exp_coords ) {
    cell.cut( dir, off );

    CHECK_REPR( td, cell.true_dimensionality() );

    // P( cell );

    Vec<Vec<double>> coords;
    for( PI n = 0; n < cell.nb_vertices_true_dim(); ++n )
        coords << cell.vertex_coord( n, td );
    CHECK_REPR( coords, exp_coords );
}

auto edges_and_rays( auto &cell ) {
    using Pt = DECAYED_TYPE_OF( cell )::Pt;
    Vec<std::tuple<PI,Pt>> rays;
    Vec<Vec<PI,2>> edges;
    cell.for_each_ray_and_edge( [&]( auto refs, PI num_vertex ) {
        rays << std::tuple<PI,Pt>{ num_vertex, cell.ray_dir( refs, num_vertex ) };
    }, [&]( auto refs, auto num_vertices ) {
        edges << num_vertices;
    } );
    return std::pair( edges, rays );
}

// Pb à regarder : intersection de rayon => il faudrait tester si le point est extérieur plutôt que regarder le signe de alpha

TEST_CASE( "Cell", "" ) {
    using C = Cell<Arch,double,3>;
    using V = Vec<double,3>;
    C cell;

    test_cut( cell, V{ +1, +0, +0 }, 0, CtInt<1>(), { { 0 } } );                                   // 1 vert(s) in 1D 
    test_cut( cell, V{ +0, +1, +0 }, 0, CtInt<2>(), { { 0, 0 } } );                                // 1 vert(s) in 2D
    test_cut( cell, V{ -1, -1, +0 }, 1, CtInt<2>(), { { 0, 0 }, { -1, 0 }, { 0, -1 } } );          // 3 vert(s) in 2D
    test_cut( cell, V{ +0, +0, +1 }, 1, CtInt<3>(), { { 0, 0, 1 }, { -1, 0, 1 }, { 0, -1, 1 } } ); // 3 vert(s) in 3D
    test_cut( cell, V{ +0, +0, -1 }, 0, CtInt<3>(), {                                              
        { 0, 0, 1 }, { -1, 0, 1 }, { 0, -1, 1 },
        { 0, 0, 0 }, { -1, 0, 0 }, { 0, -1, 0 },
    } );                                                                                           // 6 vert(s) in 3D

    auto ear = edges_and_rays( cell );
    Vec<Vec<PI>> exp{ { 0, 1 }, { 1, 2 }, { 0, 2 }, { 3, 4 }, { 4, 5 }, { 3, 5 }, { 0, 3 }, { 1, 4 }, { 2, 5 } };
    CHECK_REPR( _sorted( ear.first ), _sorted( exp ) );
}
