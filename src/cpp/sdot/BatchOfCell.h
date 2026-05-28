#pragma once

#include <sdot/generated_includes/BatchOfCell.h>

namespace sdot {

PARAMETERS_DECLARATION_OF_BatchOfCell
struct BatchOfCell {
    ATTRIBUTES_OF_BatchOfCell

    ///
    struct MeasureFunctor {
        // The per-thread scratch (map_items / nb_map_items) has map_items.shape(0) rows, indexed by
        // global_id() in per_thread(). Reporting that as the GPU worker cap makes launch_cuda_run_parallel
        // bound the grid to it (threads grid-stride over the items) so global_id() never exceeds the
        // scratch. HD (not GD) because it is queried from the host launcher.
        HD auto max_gpu_threads( auto &&map_items, auto &&.../* nb_map_items, outputs, max_nb_cuts, batch_of_cells */ ) const {
            return PI( map_items.shape( Ct<int,0>() ) );
        }

        GD void per_thread( const auto &thread_info, const auto &/* batch_indices */, auto &&cont, auto &&map_items, auto &&nb_map_items, auto &&outputs, PI max_nb_cuts, auto &&batch_of_cells ) const {
            auto item_map = recursive_map_of_unique_sorted_indices( Ct<int,ct_dim-1>(), map_items( thread_info.global_id() ), nb_map_items( thread_info.global_id() ), max_nb_cuts );
            cont( outputs, batch_of_cells, item_map );
        }

        GD void operator()( const auto &batch_index, auto outputs, auto batch_of_cells, auto item_map ) const {
            outputs( batch_index ) = batch_of_cells( batch_index ).measure( item_map );
        }
    };

};

} // namespace sdot
