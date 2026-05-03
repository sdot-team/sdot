#include <sdot/generated_includes/PolynomialGrid.h>
#include <sdot/support/TODO.h>

namespace sdot {

template<int ct_dim,typename Arch,typename TF,typename TI>
struct PolynomialGridWorker {
    using Pg = PolynomialGrid<ct_dim,Arch,TF,TI>;

    TF mass() {
        return 18;
    }


    Pg pg;
};

void mass_of_polynomial_grid( auto &&p ) {
    PolynomialGridWorker pw( p.polynomial_grid );
    p.res() = pw.mass();
}

void backward_mass_of_polynomial_grid( auto &&p ) {
    TODO;
}

}
