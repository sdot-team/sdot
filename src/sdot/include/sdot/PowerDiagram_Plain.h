#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram_Plain.h>
#include <loom/support/common_macros.h>
#include "diagram/Common.h"
#include "engine/Seeds.h"

namespace sdot {

// THE SEEDS AS THEY CAME ( `PowerDiagram_Plain.py` ): `seed_positions [ n, d ]`, `weights
// [ n ]`, and no acceleration -- each cell is cut by the `n - 1` bisectors, in
// order. The floor against which `PowerDiagram_Bsp` is measured, and what remains when the
// positions are a tracer ( no tree to build on them ).
SDOT_TEMPLATE_DECL_FOR_PowerDiagram_Plain
struct PowerDiagram_Plain {
    SDOT_ATTRIBUTES_OF_PowerDiagram_Plain
    SDOT_DIAGRAM_COMMON( seed_positions, weights )

    HD SI user_id( SI k ) const { return seed_of( k ); }

    template<class TK>
    HD auto provider( SI k0 ) const { return ProviderAll<PowerDiagram_Plain,TK,ct_dim>( *this, k0 ); }

    /// the same seeds without tensor types, for the cell engine compiled as a unit ( `engine/Cells.h` )
    HD auto engine_seeds() const {
        return engine::PlainSeeds<TF,ct_dim,has_weights>{ engine::view2<TF>( seed_positions ), engine::opt_view1<TF>( weights ) };
    }

    /// the same diagram, the weights read ELSEWHERE ( see `PowerDiagram_Bsp::with_weights` ); the order
    /// of the members is that of `PowerDiagram.py` + `PowerDiagram_Plain.py`
    /// GROUPS: the same diagram, the virtual weights of its aggregates' children read elsewhere ( what the solver writes, `sdotplan/Groups.h` )
    HD auto with_groups( auto &&vweights_, auto &&agg_vweights_ ) const {
        return ::sdot::PowerDiagram_Plain{ box_min, box_max, bnd_directions, bnd_offsets, start_vertices, start_topo, periodic_shifts, periodic_linear, sink_weight, radial_eps, nb_starts, nb_topo, nb_images, nb_points, nb_boundaries, nb_dims, group_rep, group_next, group_parent, vweights_, group_dirs, group_agg_parent, group_agg_center, group_agg_scale, group_agg_outer, agg_vweights_, nb_aggs, seed_positions, weights };
    }

    HD auto with_weights( auto &&weights_ ) const {
        return ::sdot::PowerDiagram_Plain{ box_min, box_max, bnd_directions, bnd_offsets, start_vertices, start_topo, periodic_shifts, periodic_linear, sink_weight, radial_eps, nb_starts, nb_topo, nb_images, nb_points, nb_boundaries, nb_dims, group_rep, group_next, group_parent, group_vweights, group_dirs, group_agg_parent, group_agg_center, group_agg_scale, group_agg_outer, group_agg_vweights, nb_aggs, seed_positions, weights_ };
    }
};

}
