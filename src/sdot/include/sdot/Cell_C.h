#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/Cell_C.h>
#include <loom/support/common_macros.h>
#include "cell/LocalN.h"
#include "cell/Ops.h"

namespace sdot {

// THE CANONICAL CELL ( `Cell_C.py` ): the simple polytope of `Cell_N`, in any dimension `>= 2`, cut with the CANONICAL PREDICATES
// ( `cell/Canonical.h` ) when a diagram builds it -- what `PowerDiagram( predicates = "canonical" )` starts each cell from, and what the
// cell complex glues. Same tensors as `Cell_N`, plus `vertex_ties`: per vertex, a bit per neighbor whose edge has zero length.
SDOT_TEMPLATE_DECL_FOR_Cell_C
struct Cell_C {
    SDOT_ATTRIBUTES_OF_Cell_C

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( vertex_positions )::TF;
    template<class TK> using Local = LocalN<TK,ct_dim,true>;

    HD void init_as_hypercube( auto &&scratch, auto &&origin, auto &&axes, SI cut_id ) {
        cell_ops::init_as_hypercube<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, scratch, origin, axes, cut_id );
    }
    HD void init_as_unbounded( auto &&scratch ) {
        cell_ops::init_as_unbounded<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, scratch );
    }
    HD void cut( auto &&res, auto &&scratch, auto &&direction, auto &&offset, SI cut_id ) const {
        cell_ops::cut<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, scratch, direction, offset, cut_id );
    }
    HD void measure( auto &&res, auto &&scratch ) const {
        cell_ops::measure<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, scratch );
    }
    HD void measure_bwd( auto &&res, auto &&grad_res, auto &&grad_vertex_positions, auto &&scratch ) const {
        cell_ops::measure_bwd<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, grad_res, grad_vertex_positions, scratch );
    }
};

}
