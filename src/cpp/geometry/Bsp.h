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
    using Pt = Point<TF,ct_dim>; ///< point
    using Ad = AdditionalPtData; ///<
    using Ci = std::array<PI,2>; ///< child indices
    using Ce = Cell<TF,ct_dim>;  ///< cell

    struct Node {
        /**/ Node ( PI dim ) : child_indices{ 0, 0 }, split_dir( dim ), cell( dim ) {}

        bool final() const { return child_indices[ 0 ] == 0; /* only the root can have index 0 */ }

        Ci   child_indices;
        PI   beg_pt_data;
        PI   end_pt_data;
        Pt   split_dir;
        TF   split_dot;
        Ce   cell;
    };

    struct PtData {
        Ad   additional_data;
        Pt   position;
    };

    struct AvgData {
        Pt sum;
        PI len;
    };

    struct CovData {
        Cm sum;
        PI len;
    };

    /**/                Bsp          ( PI nb_points, PI dim );

    static auto         avg_reduction( const std::vector<AvgData> &a, const std::vector<AvgData> &b ) -> std::vector<AvgData>;
    static auto         cov_reduction( const std::vector<CovData> &a, const std::vector<CovData> &b ) -> std::vector<CovData>;
    static Pt           split_dir    ( const std::vector<CovData> &cov );

    auto                avg_data_for ( TensorView<const TF,2> points ) const -> std::vector<AvgData>;
    auto                cov_data_for ( TensorView<const TF,2> points, const std::vector<AvgData> &avg ) const -> std::vector<CovData>;
    PI                  cell_number  ( Pt pos ) const;

    PI                  nb_points;   ///< will be equal to pt_data.size() at some point (but not during the construction)
    std::vector<PtData> pt_data;
    std::vector<Node>   nodes;
    PI                  dim;
};


} // namespace sdot

template<class AdditionalPtData,class TF,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim> &p );

#include "Bsp.cxx"
