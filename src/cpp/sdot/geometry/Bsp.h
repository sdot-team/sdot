#pragma once

// #include "../support/SimpleSquareMatrix.h"
#include "../support/TensorView.h"
#include "../support/P.h"
// #include "Cell.h"

namespace sdot {

template<int ct_dim,class TF>
std::pair<DsVec<TF,ct_dim>,DsVec<TF,ct_dim>> min_max( CtInt<ct_dim>, TensorView<TF,2,Cpu> pts, const auto &indices ) {
    const PI dim = pts.size( 1 );
    const PI ind = indices( 0 );
    DsVec<TF,ct_dim> mi( pts.row( ind ) );
    DsVec<TF,ct_dim> ma = mi;
    for( PI i = 1; i < pts.size( 0 ); ++i ) {
        const PI ind = indices( i );
        for( PI d = 0; d < dim; ++d ) {
            const TF v = pts( ind, d );
            mi( ind, d ) = min( mi( ind, d ), v );
            ma( ind, d ) = max( ma( ind, d ), v );
        }
    }
    return mi, ma;
}

void make_bsp( auto &&p ) {
    // sorted_vertex_indices = Return( Tensor, [ nb_vertices ], int ),
    // cell_children = Return( Tensor, [ max_nb_cells, 2 ], int ),
    // cell_bounds = Return( Tensor, [ max_nb_cells, 2 ] ),
    // nb_cells = Return( Tensor, [ max_nb_cells, 3 * dim + 1 ], int ),
    // max_points_per_cell = max_points_per_cell,
    // positions = positions,
    // weights = weights,

    //
    P( min_max( p.ct_dim ) );
}


// ///
// template<class AdditionalPtData,class TF,int ct_dim,class Arch>
// class Bsp {
// public:
//     using Cell = sdot::Cell<TF,ct_dim,Arch>;  ///< cell

//     using Cm   = DsVec<TF,(ct_dim>=0?ct_dim*(ct_dim+1)/2:-1),Arch>; ///< covariance matrix
//     using Pd   = DsVec<TF,(ct_dim>=0?ct_dim+1:-1),Arch>; ///< point + 1 item
//     using Pf   = DsVecFactory<TF,ct_dim,Arch>; ///<
//     using Pt   = DsVec<TF,ct_dim,Arch>; ///< point
//     using Ad   = AdditionalPtData; ///<
//     using Ci   = std::array<PI,2>; ///< child indices
//     using Sm   = std::array<TF,2>; ///< split maximums

//     enum class NodeType { Split, Final, Ext };

//     struct SplitData {
//         Ci   child_indices;
//     };

//     struct FinalData {
//         PI   nb_points() const { return end_pt_data - beg_pt_data; }

//         PI   beg_pt_data;
//         PI   end_pt_data;
//     };

//     struct ExtData {
//         PI   num_item;
//     };

//     struct Node {
//         NodeType type;
//         union {
//             SplitData split;
//             FinalData final;
//             ExtData   ext;
//         } data;
//     };

//     struct PtData {
//         Ad   additional_data;
//         Pt   position;
//         TF   weight;
//         PI   index;
//     };

//     struct AvgData {
//         Pt sum;
//         PI len;
//     };

//     struct CovData {
//         Cm sum;
//         PI len;
//     };

//     /**/                Bsp               ( const auto &node_summary, PI node_index, TensorView<const TF,2,Arch> positions, TensorView<const PI,1,Arch> indices, TensorView<const TF,1,Arch> weights, PI max_points_per_cell );

//     bool                is_in_charge_of   ( const Pt &pos ) const;
//     void                display_vtk       ( VtkOutput &vo ) const;

//     void                update_children_of( PI node_index, PI max_points_per_cell );
//     Cell                make_new_cell     ( const Cell &base_cell, const Pt &split_dir, PI beg, PI end );
//     auto                split_hst_for     ( TensorView<const TF,2,Arch> points, const Pt &split_dir, TF split_beg, TF split_end, PI nb_bins ) const -> std::vector<TF>;
//     void                display_rec       ( std::ostream &os, PI node_index, std::string prefix = "" ) const;
//     void                add_path          ( TensorView<const TF,2,Arch> path, PI num_bsp );

//     void                for_each_cell     ( const auto &primitive, const auto &sorted_potentials, auto &&func );
//     auto                remake_cell       ( const auto &cell, const auto &primitive, const auto &sorted_potentials );

//     friend std::ostream &operator<<       ( std::ostream &os, const Bsp &p ) { p.display_rec( os, 0 ); return os; }

//     PI                  nb_points;        ///< will be equal to pt_data.size() at some point (but not during the construction)
//     std::vector<PtData> pt_data;
//     std::vector<Node>   nodes;
//     std::vector<Cell>   cells;
//     PI                  dim;
//     Pf                  pf;
// };

} // namespace sdot

// template<class AdditionalPtData,class TF,int dim,class Arch>
// friend std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim,Arch> &p );

// #include "Bsp.cxx"
