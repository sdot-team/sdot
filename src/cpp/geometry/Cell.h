#pragma once

#include "RecursiveMapOfUniqueSortedIndices.h"
#include "SimpleSquareMatrix.h"
#include "PointFactory.h"
#include "VtkOutput.h"

namespace sdot {

//
template< class TF, int ct_dim = -1 >
class Cell {
public:
    struct      FaceCorr { PI vertex_index_plus_curr_op_id = 0; PI cut_ind_to_remove; };
    struct      ItemCorr { PI vertex_index_plus_curr_op_id = 0; };

    using       FaceMap  = MapOfUniqueSortedIndices<(ct_dim>1?ct_dim-2:-1),PI,FaceCorr>;
    using       ItemMap  = RecursiveMapOfUniqueSortedIndices<ct_dim,PI,ItemCorr>;
    using       PF       = PointFactory<TF,ct_dim>;
    using       DF       = PointFactory<PI,ct_dim>;
    using       VF       = std::vector<TF>;
    using       VI       = std::vector<int>;
    using       Pt       = Point<TF,ct_dim>;
    using       It       = Point<PI,ct_dim>;

    struct EdgeLink {
        PI vertex_index;
        PI num_cut_to_remove;
    };
    struct Vertex {
        using     EdgeLinks = std::vector<EdgeLink>;

        Pt        pos;
        It        cut_indices;
        EdgeLinks edge_links;
    };
    struct Cut {
        Pt   dir;
        TF   sp;
        PI   id;
        bool ext;
    };

    using           Vertices                      = std::vector<Vertex>;
    using           Cuts                          = std::vector<Cut>;

    /**/            Cell                          ( int actual_dim );

    void            init_with_axis_aligned_simplex( TF length );
    void            init_with_englobing_simplex   ( TF radius );
    void            init_with_simplex             ( std::span<Pt> points );

    void            check_consistency             ( TF eps = 1e-6 ) const;
    void            display_vtk                   ( VtkOutput &vo ) const;
    TF              measure                       () const;
    PI              dim                           () const { return pf.dim(); }

    void            cut                           ( const Pt &dir, TF sp, PI id );

    static void     _for_each_2_comb_excepted     ( PI size, PI excepted, auto&& func );
    void            _cut_int_ext_edge             ( PI n0, EdgeLink &e0, TF s0, TF s1 );
    void            _add_measure_rec              ( TF &res, SimpleSquareMatrix<TF,ct_dim> &M, const auto &cut_indices, PI prev_vertex_index ) const;

    VI              vertex_corr;                  ///<
    mutable PI64    curr_op_id;                   ///<
    VI              cut_corr;                     ///<
    FaceMap         face_map;                     ///<
    mutable ItemMap item_map;                     ///<
    Vertices        vertices;                     ///<
    Cuts            cuts;                         ///<
    VF              sps;                          ///< scalar product
    PF              pf;                           ///< point factory
    DF              df;                           ///< index factory
};

//
// template<class T>
// class Cell<T,2> {
// public:

// };

} // namespace sdot

T_Td std::ostream& operator<<( std::ostream& os, const sdot::Cell< T, d >& p );

#include "Cell.cxx"
