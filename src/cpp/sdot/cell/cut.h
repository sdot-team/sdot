#pragma once

#include <sdot/generated_includes/CutWorkspace.h>
#include <sdot/generated_includes/Cell.h>

namespace sdot {

template<int ct_dim,class Arch,class TF,class TI>
struct CellWorker {
    using Pt = DsVec<TF,ct_dim,Arch>; ///< point
    using Ci = DsVec<TI,ct_dim,Arch>; ///< cut indices

    //
    Pt    vertex_position       ( PI num_vertex ) const;
    Ci    vertex_indices        ( PI num_vertex ) const;
    bool  vertex_inf            ( PI num_vertex ) const; ///< true if at least one cut is inf
    Pt    cut_dir               ( PI num_cut ) const;
    TF    cut_dot               ( PI num_cut ) const;

    Pt    solve_position        ( PI num_vertex, auto &&add_func ) const;
    Pt    solve_position        ( PI num_vertex ) const;

    //
    void  remove_unused_vertices( PI nb_vertices_orig );
    void  remove_unused_edges   ();
    void  remove_unused_cuts    ();

    void  apply_vertex_corr     ();
    void  apply_cut_corr        ();

    void  check_if_fully_closed ();
    PI    register_the_new_cut  ( const auto &cut_dir, auto cut_dot, SI cut_id );
    void  grow_infinite_cuts    ( const auto &cut_dir, auto cut_dot );
    PI    scalar_products       ( const auto &cut_dir, auto cut_dot );
    void  process_edges         ( PI nc );
    void  swap_and_pop          ( auto &nb, auto &&move_row ); ///< generic swap-and-pop (indices_to_remove sorted ascending), fills ws.corr with old->new map

    void  check_consistency     ();
    void  clear_cell            ();
    void  disp_cell             ();

    void  cut_2d                ( const auto &cut_dir, auto cut_dot, SI cut_id, PI nb_out );
    void  cut                   ( const auto &cut_dir, auto cut_dot, SI cut_id );

    Cell<ct_dim,Arch,TF,TI>&   cell;
    CutWorkspace<Arch,TF,TI>&  ws;
    const PI                   dim;
};

template<int ct_dim,class Arch,class TF,class TI>
void cut( Cell<ct_dim,Arch,TF,TI> &cell, CutWorkspace<Arch,TF,TI> &ws, const auto &cut_dir, auto cut_dot, SI cut_id ) {
    CellWorker<ct_dim,Arch,TF,TI> cw( cell, ws, cell.dim() );
    cw.cut( cut_dir, cut_dot, cut_id );
}

} // namespace sdot

#include "cut.cxx"
