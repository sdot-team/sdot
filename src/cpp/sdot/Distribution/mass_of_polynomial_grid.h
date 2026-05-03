#include <sdot/generated_includes/PolynomialGrid.h>
#include <sdot/Distribution/PolynomialGridWorker.h>
#include <sdot/support/TODO.h>
#include <sdot/support/P.h>

namespace sdot {

void mass_of_polynomial_grid( auto &&p ) {
    with_worker_for( p.polynomial_grid, [&]( auto &pw ) {
        p.res() = pw.mass();
    } );
}

void backward_mass_of_polynomial_grid( auto &&p ) {
    TODO;
}

}
