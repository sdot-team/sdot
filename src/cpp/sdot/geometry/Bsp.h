#pragma once

#include "../support/TensorView.h"
#include "SimpleSquareMatrix.h"
#include "Cell.h"

namespace sdot {

///
template<class AdditionalPtData,class TF,int ct_dim,class Arch>
class Bsp {
public:
    using Cm = Point<TF,(ct_dim>=0?ct_dim*(ct_dim+1)/2:-1),Arch>; ///< covariance matrix
    using Pd = Point<TF,(ct_dim>=0?ct_dim+1:-1),Arch>; ///< point + 1 item
    using Pf = PointFactory<TF,ct_dim,Arch>; ///<
    using Pt = Point<TF,ct_dim,Arch>; ///< point
    using Ce = Cell<TF,ct_dim,Arch>;  ///< cell
    using Ad = AdditionalPtData; ///<
    using Ci = std::array<PI,2>; ///< child indices
    using Sm = std::array<TF,2>; ///< split maximums

    struct Node {
        /**/ Node     ( PI dim ) : child_indices{ 0, 0 }, beg_pt_data( 0 ), end_pt_data( 0 ), split_dir( dim ), cell( dim ) {}

        PI   nb_points() const { return end_pt_data - beg_pt_data; }
        bool final    () const { return child_indices[ 0 ] == 0; /* only the root can have index 0 */ }

        Ci   child_indices;
        PI   beg_pt_data;
        PI   end_pt_data;
        Sm   split_maxs;
        Pt   split_dir;
        TF   split_dot;
        PI   num_bsp;
        bool local;
        Ce   cell;
    };

    struct PtData {
        Ad   additional_data;
        Pt   position;
        PI   index;
    };

    struct AvgData {
        Pt sum;
        PI len;
    };

    struct CovData {
        Cm sum;
        PI len;
    };

    /**/                Bsp            ( TensorView<const TF,3,Arch> all_the_paths, TensorView<const TF,2,Arch> min_max, TensorView<const PI,1,Arch> local_indices, TensorView<const TF,2,Arch> local_points,
                                         TensorView<const TF,2,Arch> local_path, PI max_points_per_cell );

    bool                is_in_charge_of( const Pt &pos ) const;
    void                display_vtk    ( VtkOutput &vo ) const;

    void                make_node_cells( PI node_index );
    auto                split_hst_for  ( TensorView<const TF,2,Arch> points, const Pt &split_dir, TF split_beg, TF split_end, PI nb_bins ) const -> std::vector<TF>;
    auto                sum_pos_for    ( TensorView<const TF,2,Arch> points ) const -> Pt; ///< [ sum of xs, ..., sum of zs, sum of 1 ]
    auto                sum_cov_for    ( TensorView<const TF,2,Arch> points, const Pt &avg ) const -> SimpleSquareMatrix<TF,-1,Arch>;
    void                display_rec    ( std::ostream &os, PI node_index, std::string prefix = "" ) const;
    PI                  cell_number    ( Pt pos ) const;
    void                fill_node      ( PI node_index, const PI beg_pt_data, const PI end_pt_data, PI max_points_per_cell );
    void                add_path       ( TensorView<const TF,2,Arch> path, PI num_bsp );

    PI                  nb_points;     ///< will be equal to pt_data.size() at some point (but not during the construction)
    std::vector<PtData> pt_data;
    std::vector<Node>   nodes;
    PI                  dim;
    Pf                  pf;
};

        // split_hst_for_each_chunk = [ dask.delayed( bsp.value.split_hst_for )( chunk.value, split_dir,  ) for chunk in delayed_chunks ]

} // namespace sdot

template<class AdditionalPtData,class TF,int dim,class Arch>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim,Arch> &p );

#include "Bsp.cxx"
