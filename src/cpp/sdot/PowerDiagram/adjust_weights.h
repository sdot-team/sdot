#include "../Distribution/PolynomialGridWorker.h"
#include <sdot/generated_includes/MatrixTerms.h>
#include "../support/LinearSolver.h"
#include "../support/dichotomy.h"
#include "PowerDiagramWorker.h"
#include "../support/P.h"
#include <iomanip>

#include <cstdlib>

namespace sdot {

template<typename Arch, typename _TF, typename TI>
struct MatrixTermsWorker {
    using TF = _TF;

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
int get_matrix_terms( auto &p, auto &&distribution_worker, MatrixTermsWorker<Arch,TF,TI> &mw ) {
    using namespace std;

    PowerDiagramWorker pw( p.power_diagram, p.cell_workspace, p.cells );
    mw.clear();

    mw.mt.max_measure_error_ratio() = 0;

    int error = 0;
    distribution_worker.with_preparation_for_cell_traversal( p.cell_workspace, p.cells, [&]( auto &distribution_worker ) {
        error = pw.for_each_cell( distribution_worker, [&]( auto &cell_worker, PI batch_index, SI num_dirac_0 ) {
            auto d0_center = pw.position( num_dirac_0 );
            TF der_0 = 0;
            TF mass = 0;
            distribution_worker.for_each_sub_cell( cell_worker.cell, batch_index, [&]( auto &cell_worker, const auto &local_function ) {
                mass += integral( local_function, cell_worker );

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

            if ( mass == 0 )
                return 1;

            mw.mt.vector_vals[ num_dirac_0 ] = p.dirac_masses[ num_dirac_0 ] - mass;
            mw.add_term( num_dirac_0, num_dirac_0, der_0 );

            mw.mt.max_measure_error_ratio() = max( mw.mt.max_measure_error_ratio(), abs( p.dirac_masses[ num_dirac_0 ] - mass ) / mass );

            return 0;
        } );
    } );

    if ( p.matrix_terms.nb_matrix_terms ) {
        mw.add_term( 0, 0, 1 );
    }

    return error;
}

enum {
    NO_ERROR = 0,
    ERROR_ON_FIRST_SYSTEM,
    ERROR_MAX_ITER,
};

int newton_with_backtracking( auto &p ) {
    int error = 0;
    with_worker_for( p.target_distribution, [&]( auto &&distribution ) {
        MatrixTermsWorker mw( p.matrix_terms );
        for( PI num_iter = 0; ; ) {
            int system_error = get_matrix_terms( p, distribution, mw );
            if ( system_error ) {
                if ( num_iter == 0 ) {
                    error = ERROR_ON_FIRST_SYSTEM;
                    return;
                }

                using TF = DECAYED_TYPE_OF( p.matrix_terms.solution[ 0 ] );
                const TF coeff_backtracking = 0.5;
                for( PI n = 0; n < p.matrix_terms.solution.size(); ++n ) {
                    p.matrix_terms.solution[ n ] *= coeff_backtracking;
                    p.power_diagram.weights[ n ] -= p.matrix_terms.solution[ n ];
                }
                continue;
            }

            if ( p.verbosity )
                std::cout << "niter: " << std::setw( 10 ) << num_iter << " merr: " << std::setw( 10 ) << std::scientific << mw.mt.max_measure_error_ratio() << std::endl;

            if ( mw.mt.max_measure_error_ratio() < 1e-3 )
                return;

            if( ++num_iter == p.max_iteration_count ) {
                error = ERROR_MAX_ITER;
                break;
            }

            mw.solve();

            for( PI n = 0; n < p.matrix_terms.solution.size(); ++n )
                p.power_diagram.weights[ n ] += p.matrix_terms.solution[ n ];

        }
    } );

    return error;
}

void center_weights( auto &p ) {
    using namespace std;

    constexpr PI dim = DECAYED_TYPE_OF( p.power_diagram.ct_dim )::value;
    using TF = DECAYED_TYPE_OF( p.new_weights[ 0 ] );
    using Pt = Vector<TF,Arch,dim,Cpu>;

    with_worker_for( p.target_distribution, [&]( auto &&distribution_worker ) {
        distribution_worker.with_preparation_for_cell_traversal( p.cell_workspace, p.cells, [&]( auto &distribution_worker ) {
            auto cell_workspace = p.cell_workspace.row( 0 );
            auto cell = p.cells.row( 0 );
            CellWorker cw( cell, cell_workspace, cell.dim() );
            distribution_worker.init_cell( cell );

            // find a (hyper) cube that fits inside the cell
            auto cube_center = cw.centroid();
            auto has_outside = [&]( TF sca ) {
                for( PI d = 0; d < cw.dim; ++d ) {
                    for( TF v : { -sca, +sca } ) {
                        auto p = cube_center;
                        p[ d ] += v;

                        if ( ! cw.contains( p ) )
                            return true;
                    }
                }
                return false;
            };

            TF cube_width = dichotomy_find_largest_zero( TF( 0 ), has_outside ) * 0.9;

            // min max diracs
            Pt mi_di = p.power_diagram.positions.row( 0 );
            Pt ma_di = mi_di;
            for( PI n = 1; n < p.power_diagram.positions.size( 0 ); ++n ) {
                for( PI d = 0; d < dim; ++d ) {
                    mi_di[ d ] = min( mi_di[ d ], p.power_diagram.positions( n, d ) );
                    ma_di[ d ] = max( ma_di[ d ], p.power_diagram.positions( n, d ) );
                }
            }
            Pt diracs_center = ( ma_di + mi_di ) / 2;
            Pt diracs_width = ( ma_di - mi_di ) / 2;

            // adjust the weights
            Pt scal = Pt::with_func( dim, [&]( PI d ) { return cube_width / diracs_width[ d ] - 1; } );
            Pt grad = Pt::with_func( dim, [&]( PI d ) { return 2 * ( diracs_center[ d ] - cube_center[ d ] ); } );
            for( PI n = 0; n < p.power_diagram.positions.size( 0 ); ++n ) {
                TF w = dot( grad, Pt( p.power_diagram.positions.row( n ) ) );
                for( PI d = 0; d < dim; ++d )
                    w -= scal[ d ] * pow( p.power_diagram.positions( n, d ) - diracs_center[ d ], 2 ) / 2;
                p.power_diagram.weights[ n ] = w;
            }
        } );
    } );
}

int adjust_weights( auto &&p ) {
    if ( p.power_diagram.weights.is_valid() )
        p.power_diagram.weights.spill_to( p.new_weights );
    else
        TODO;

    center_weights( p );

    bool weights_has_been_adjusted_to_fit_density_boundaries = false;
    while ( true ) {
        int error = newton_with_backtracking( p );
        if ( error == 0 )
            return 0;

        if ( error == ERROR_ON_FIRST_SYSTEM ) {
            if ( weights_has_been_adjusted_to_fit_density_boundaries )
                TODO;
            weights_has_been_adjusted_to_fit_density_boundaries = true;

            center_weights( p );
            continue;
        }

        if ( error == ERROR_MAX_ITER ) {
            return ERROR_MAX_ITER;
        }

        TODO;
    }
}

void adjust_weights_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
