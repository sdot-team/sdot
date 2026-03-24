#include "../src/cpp/geometry/Cell.h"
#include "../src/cpp/support/P.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;
using TF = double;

template<int ct_dim=-1>
void test_sum_area_hypercube( PI dim = ct_dim ) {
    auto base_cell = Cell<TF,ct_dim>::axis_aligned_hypercube( dim, 1 );
    ASSERT( abs( base_cell.measure() - 1 ) < 1e-5 );
    base_cell.check_consistency();

    std::vector<Cell<TF,ct_dim>> cells;
    cells.push_back( base_cell );

    for( PI d = 0; d < dim; ++d ) {
        TF tot_measure = 0;
        std::vector<Cell<TF,ct_dim>> new_cells;
        for( const auto &cell : cells ) {
            Cell<TF,ct_dim> n0 = cell;
            Cell<TF,ct_dim> n1 = cell;
            n0.cut( n0.pf.value_at( d, +1 ), 0, d );
            n1.cut( n1.pf.value_at( d, -1 ), 0, d );

            ASSERT( abs( n0.measure() - pow( 0.5, d + 1 ) ) < 1e-5 );
            ASSERT( abs( n1.measure() - pow( 0.5, d + 1 ) ) < 1e-5 );
            tot_measure += n0.measure();
            tot_measure += n1.measure();

            new_cells.push_back( std::move( n0 ) );
            new_cells.push_back( std::move( n1 ) );
        }

        ASSERT( abs( tot_measure - 1 ) < 1e-5 );
        cells = std::move( new_cells );
    }

    VtkOutput vo;
    base_cell.display_vtk( vo );
    vo.save( "build/out.vtk" );
}

TEST_CASE("2D power diagram", "[PD]") {
    // test_sum_area_hypercube( 3 );
    // SECTION( "1" ) {
    //     Cell<TF,2> p2( 2 );
    //     P( p2 );

    //     p2.cut( { 1, 0 }, 0.7, 17 );
    //     P( p2 );
    // }

    // SECTION( "2" ) {
    //     Cell<TF,2> p2( 2 );
    //     P( p2 );

    //     p2.cut( { 0, 1 }, 0.7, 17 );
    //     P( p2 );
    // }

    // SECTION( "3" ) {
    //     Cell<TF,2> p2( 2 );
    //     P( p2 );

    //     p2.cut( { -1, -1 }, -0.3, 17 );
    //     P( p2 );
    // }

    // SECTION( "square" ) {
    //     Cell<TF,2> p2 = Cell<TF,2>::axis_aligned_hypercube( { -1, -1 }, { +1, +1 } );
    //     P( p2 );

    //     p2.cut( { 1, 0 }, 0.5, 17 );
    //     P( p2 );
    // }

    SECTION( "circle" ) {
        for( PI nb = 5, cpt = 0; nb < 360; nb += 15, ++cpt ) {
            Cell<TF,2> p2 = Cell<TF,2>::axis_aligned_hypercube( { -10, -10 }, { +10, +10 } );

            for( PI na = 0; na < 360; na += 20 ) {
                const TF a = M_PI * na / 180;
                p2.cut( { cos( a ), sin( a ) }, 1, 17 );
            }

            const TF b = M_PI * nb / 180;
            p2.cut( { cos( b ), sin( b ) }, 0.3, 17 );

            VtkOutput vo;
            p2.display_vtk( vo );
            vo.save( "build/out_" + std::to_string( cpt ) + ".vtk" );
        }
    }

    // p2.init_with_axis_aligned_simplex( 2.0 );
    // P( p2 );
    // P( p2.measure() );

    // p2.cut( p2.pf( 1, 0 ), 1.0, 17 );
    // P( p2 );
    // P( p2.measure() );

    // p2.cut( p2.pf( -1, 0 ), -0.5, 17 );
    // P( p2 );
    // P( p2.measure() );

    // Cell<TF,3> p3( 3, 10.0 );
    // P( p3 );

    // p3.cut( p3.pf( 1, 0, 0 ), 1, 17 );
    // P( p3 );

}
