#include "../Distribution/PolynomialGridWorker.h"
#include <sdot/generated_includes/MatrixTerms.h>
#include "../support/LinearSolver.h"
#include "PowerDiagramWorker.h"
#include "../support/P.h"

namespace sdot {

template<typename Arch, typename TF, typename TI>
struct MatrixTermsWorker {
    void clear() {
        mt.nb_matrix_terms = 0;
    }

    void add_term( TI r, TI c, TF v ) {
        PI p = mt.nb_matrix_terms++;
        mt.matrix_rows[ p ] = r;
        mt.matrix_cols[ p ] = c;
        mt.matrix_vals[ p ] = v;
    }

    void solve() {
        LinearSolver<Arch, TF, TI>::solve_spd(
            mt.vector_vals.size(),
            mt.nb_matrix_terms,
            mt.matrix_rows.data(),
            mt.matrix_cols.data(),
            mt.matrix_vals.data(),
            mt.vector_vals.data(),
            mt.solution.data()
        );
    }

    MatrixTerms<Arch,TF,TI> &mt;
};

template<typename Arch, typename TF, typename TI>
void get_matrix_terms( auto &p, auto &&distribution_worker, MatrixTermsWorker<Arch,TF,TI> &mw ) {
    PowerDiagramWorker pw( p.power_diagram, p.cell_workspace, p.cells );
    mw.clear();

    distribution_worker.with_preparation_for_cell_traversal( p.cell_workspace, p.cells, [&]( auto &distribution_worker ) {
        pw.for_each_cell( distribution_worker, [&]( auto &cell_worker, PI batch_index, SI num_dirac_0 ) {
            auto d0_center = pw.position( num_dirac_0 );
            auto V = p.dirac_masses[ num_dirac_0 ];
            auto der_0 = V * 0;
            distribution_worker.for_each_sub_cell( cell_worker.cell, batch_index, [&]( auto &cell_worker, const auto &local_function ) {
                V -= integral( local_function, cell_worker );

                cell_worker.for_each_facet( [&]( auto &&facet, SI num_dirac_1 ) {
                    if ( num_dirac_1 < 0 )
                        return;
                    auto boundary_measure = integral( local_function, facet );
                    auto d1_center = pw.position( num_dirac_1 );

                    if ( num_dirac_0 == num_dirac_1 ) {
                        //der_0 += coeff * boundary_measure / sqrt( d0_weight );
                    } else {
                        // TI m_num_dirac_1 = num_dirac_1 % nb_diracs, d_num_dirac_1 = num_dirac_1 / nb_diracs;
                        // Pt d1_center = grid.sym( positions[ m_num_dirac_1 ], int( d_num_dirac_1 ) - 1 );
                        //if ( std::size_t nu = num_dirac_1 / nb_diracs )
                        //    TODO; // d1_center = transformation( _tranformations[ nu - 1 ], d1_center );
                        if ( auto dist = norm_2( d0_center - d1_center ) ) {
                            auto b_der = 0.5 * boundary_measure / dist;
                            mw.add_term( num_dirac_0, num_dirac_1, - b_der );
                            der_0 += b_der;
                        }
                    }
                } );

                // der_0 += cp.integration_der_wrt_weight( space_func, radial_func.func_for_final_cp_integration(), d0_weight );
            } );
            mw.add_term( num_dirac_0, num_dirac_0, der_0 );
            mw.mt.vector_vals[ num_dirac_0 ] = V;
        } );
    } );

    if ( p.matrix_terms.nb_matrix_terms ) {
        mw.add_term( 0, 0, 1 );
    }
}

void adjust_weights( auto &&p ) {
    with_worker_for( p.target_distribution, [&]( auto &&distribution ) {
        MatrixTermsWorker mw( p.matrix_terms );
        for( PI i = 0; i < 3; ++i ) {
            get_matrix_terms( p, distribution, mw );
            mw.solve();

            P( p.matrix_terms.vector_vals );
            for( PI n = 0; n < p.matrix_terms.solution.size(); ++n )
                p.power_diagram.weights[ n ] += p.matrix_terms.solution[ n ];
        }
    } );
}

void adjust_weights_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
