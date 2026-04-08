#pragma once

#include "../support/TensorView.h"
#include "SimpleSquareMatrix.h"
#include "Cell.h"

namespace sdot {

///
template<class AdditionalPtData,class TF,int ct_dim,class Arch>
class Bsp {
public:
    using Cell = sdot::Cell<TF,ct_dim,Arch>;  ///< cell

    using Cm   = Point<TF,(ct_dim>=0?ct_dim*(ct_dim+1)/2:-1),Arch>; ///< covariance matrix
    using Pd   = Point<TF,(ct_dim>=0?ct_dim+1:-1),Arch>; ///< point + 1 item
    using Pf   = PointFactory<TF,ct_dim,Arch>; ///<
    using Pt   = Point<TF,ct_dim,Arch>; ///< point
    using Ad   = AdditionalPtData; ///<
    using Ci   = std::array<PI,2>; ///< child indices
    using Sm   = std::array<TF,2>; ///< split maximums

    enum class NodeType { Split, Final, Ext };

    struct SplitData {
        Ci   child_indices;
    };

    struct FinalData {
        PI   nb_points() const { return end_pt_data - beg_pt_data; }

        PI   beg_pt_data;
        PI   end_pt_data;
    };

    struct ExtData {
        PI   num_item;
    };

    struct Node {
        NodeType type;
        union {
            SplitData split;
            FinalData final;
            ExtData   ext;
        } data;
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

    /**/                Bsp               ( const auto &node_summary, PI node_index, TensorView<const TF,2,Arch> positions, TensorView<const PI,1,Arch> indices, PI max_points_per_cell );

    bool                is_in_charge_of   ( const Pt &pos ) const;
    void                display_vtk       ( VtkOutput &vo ) const;

    void                update_children_of( PI node_index, PI max_points_per_cell );
    Cell                make_new_cell     ( const Cell &base_cell, const Pt &split_dir, PI beg, PI end );
    auto                split_hst_for     ( TensorView<const TF,2,Arch> points, const Pt &split_dir, TF split_beg, TF split_end, PI nb_bins ) const -> std::vector<TF>;
    void                display_rec       ( std::ostream &os, PI node_index, std::string prefix = "" ) const;
    void                add_path          ( TensorView<const TF,2,Arch> path, PI num_bsp );

    void                for_each_cell     ( const auto &primitive, auto &&func );

    friend std::ostream &operator<<       ( std::ostream &os, const Bsp &p ) { p.display_rec( os, 0 ); return os; }

    PI                  nb_points;     ///< will be equal to pt_data.size() at some point (but not during the construction)
    std::vector<PtData> pt_data;
    std::vector<Node>   nodes;
    std::vector<Cell>   cells;
    PI                  dim;
    Pf                  pf;
};

} // namespace sdot

// template<class AdditionalPtData,class TF,int dim,class Arch>
// friend std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim,Arch> &p );

#include "Bsp.cxx"
