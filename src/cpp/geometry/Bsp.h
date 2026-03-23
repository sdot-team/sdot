#pragma once

#include "../support/TensorView.h"
#include "SimpleSquareMatrix.h"
#include "Cell.h"

namespace sdot {

///
template<class AdditionalPtData,class TF,int ct_dim=-1>
class Bsp {
public:
    using Cm = Point<TF,(ct_dim>=0?ct_dim*(ct_dim+1)/2:-1)>; ///< covariance matrix
    using Pd = Point<TF,(ct_dim>=0?ct_dim+1:-1)>; ///< point + 1 item
    using Pt = Point<TF,ct_dim>; ///< point
    using Ad = AdditionalPtData; ///<
    using Ci = std::array<PI,2>; ///< child indices
    using Ce = Cell<TF,ct_dim>;  ///< cell

    struct Node {
        /**/ Node     ( PI dim ) : child_indices{ 0, 0 }, beg_pt_data( 0 ), end_pt_data( 0 ), split_dir( dim ), cell( dim ) {}

        PI   nb_points() const { return end_pt_data - beg_pt_data; }
        bool final    () const { return child_indices[ 0 ] == 0; /* only the root can have index 0 */ }

        Ci   child_indices;
        PI   beg_pt_data;
        PI   end_pt_data;
        Pt   split_dir;
        TF   split_dot;
        PI   num_bsp;
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

    /**/                Bsp            ( TensorView<const TF,3> all_the_paths, TensorView<const PI,1> indices, TensorView<const TF,2> points, TensorView<const TF,2> path, PI max_points_per_cell );

    bool                is_in_charge_of( const Pt &pos ) const;

    auto                split_hst_for  ( TensorView<const TF,2> points, const Pt &split_dir, TF split_beg, TF split_end, PI nb_bins ) const -> std::vector<TF>;
    auto                sum_pos_for    ( TensorView<const TF,2> points ) const -> Pt; ///< [ sum of xs, ..., sum of zs, sum of 1 ]
    auto                sum_cov_for    ( TensorView<const TF,2> points, const Pt &avg ) const -> SimpleSquareMatrix<TF>;
    PI                  cell_number    ( Pt pos ) const;
    void                display_rec    ( std::ostream &os, PI node_index, std::string prefix = "" ) const;
    void                fill_node      ( PI node_index, PI beg_pt_data, PI end_pt_data, PI max_points_per_cell );
    void                add_path       ( TensorView<const TF,2> path, PI num_bsp );

    PI                  nb_points;     ///< will be equal to pt_data.size() at some point (but not during the construction)
    std::vector<PtData> pt_data;
    std::vector<Node>   nodes;
    PI                  dim;
};

        // split_hst_for_each_chunk = [ dask.delayed( bsp.value.split_hst_for )( chunk.value, split_dir,  ) for chunk in delayed_chunks ]

} // namespace sdot

template<class AdditionalPtData,class TF,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim> &p );

#include "Bsp.cxx"
