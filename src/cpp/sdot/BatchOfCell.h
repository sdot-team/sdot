#pragma once

#include <sdot/generated_includes/BatchOfCell.h>

namespace sdot {

PARAMETERS_DECLARATION_OF_BatchOfCell
struct BatchOfCell {
    ATTRIBUTES_OF_BatchOfCell

    ///
    struct MeasureFunctor {
        GD void per_thread( const auto &thread_info, const auto &/* batch_indices */, auto &&cont, auto &&map_items, auto &&nb_map_items, auto &&outputs, PI max_nb_cuts, auto &&batch_of_cells ) const {
            auto nb_map_items_loc = nb_map_items( thread_info.global_id() );
            nb_map_items_loc = 0;

            auto item_map = recursive_map_of_unique_sorted_indices( Ct<int,ct_dim-1>(), map_items( thread_info.global_id() ), nb_map_items_loc, max_nb_cuts );

            cont( outputs, batch_of_cells, item_map );
        }

        GD void operator()( const auto &batch_index, auto outputs, auto batch_of_cells, auto item_map ) const {
            outputs( batch_index ) = batch_of_cells( batch_index ).measure( item_map );
        }
    };

};

} // namespace sdot
