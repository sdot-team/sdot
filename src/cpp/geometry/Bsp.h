#pragma once

#include "../support/TensorView.h"
#include "SimpleSquareMatrix.h"
#include "Cell.h"

namespace sdot {

///
template<class AdditionalPtData,class TF,int ct_dim=-1>
class Bsp {
public:
    using CovData = Point<TF,(ct_dim>=0?ct_dim*(ct_dim+1)/2:-1)>;
    using AvgData = Point<TF,ct_dim>;
    using Pt = Point<TF,ct_dim>;

    struct Node {
        /**/            Node( PI dim ) : mean_points( dim ), split_dir( dim ), cell( dim ) {}

        PI              child_indices;
        PI              beg_pt_data;
        PI              end_pt_data;
        Pt              mean_points;
        Pt              split_dir;
        TF              split_sp;
        Cell<TF,ct_dim> cell;
    };

    struct PtData {
        AdditionalPtData additional_data;
        Pt               position;
    };

    /**/                Bsp         ( PI dim );

    AvgData             avg_data_for( TensorView<const TF,2> points ) const;
    CovData             cov_data_for( TensorView<const TF,2> points ) const;

    std::vector<PtData> pt_data;
    std::vector<Node>   nodes;
    PI                  dim;
};


} // namespace sdot

template<class AdditionalPtData,class TF,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim> &p );

#include "Bsp.cxx"
