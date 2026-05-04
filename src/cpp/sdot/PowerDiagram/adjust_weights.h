#include "../Distribution/PolynomialGridWorker.h"
#include "PowerDiagramWorker.h"
#include "../support/P.h"

namespace sdot {

void get_matrix_terms( auto &p, auto &&distribution_worker ) {
    PowerDiagramWorker pw( p.power_diagram, p.cell_workspace, p.cells );
    double tot = 0;
    pw.for_each_cell( distribution_worker, [&]( auto &cell_worker, PI /* batch_index */ ) {
        tot += cell_worker.measure();
    } );

    P( tot );
}

void adjust_weights( auto &&p ) {
    with_worker_for( p.target_distribution, [&]( auto &&distribution ) {
        get_matrix_terms( p, distribution );
    } );
}

void adjust_weights_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
