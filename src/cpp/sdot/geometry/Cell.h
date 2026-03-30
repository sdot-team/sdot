#pragma once

#include "RecursiveMapOfUniqueSortedIndices.h"
#include "SimpleSquareMatrix.h"
#include "PointFactory.h"
#include "VtkOutput.h"

namespace sdot {

//
template<class TF,int ct_dim,class Arch>
class Cell {
public:
    struct      FaceCorr { PI64 vertex_index_plus_curr_op_id = 0; PI cut_ind_to_remove; };
    struct      ItemCorr { PI64 vertex_index_plus_curr_op_id = 0; };

    using       FaceMap  = MapOfUniqueSortedIndices<(ct_dim>1?ct_dim-2:-1),PI,FaceCorr>;
    using       ItemMap  = RecursiveMapOfUniqueSortedIndices<ct_dim,PI,ItemCorr>;
    using       PF       = PointFactory<TF,ct_dim,Arch>;
    using       DF       = PointFactory<PI,ct_dim,Arch>;
    using       Pt       = Point<TF,ct_dim,Arch>;
    using       It       = Point<PI,ct_dim,Arch>;
    using       VF       = std::vector<TF>;
    using       VI       = std::vector<int>;

    struct EdgeLink {
        PI num_cut_to_remove;
        PI vertex_index;
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

    /**/            Cell                          ( int dim );

    static Cell     axis_aligned_hypercube        ( Pt p0, Pt p1 );
    static Cell     axis_aligned_simplex          ( int dim, TF length );
    static Cell     englobing_simplex             ( Pt center, TF radius );
    static Cell     simplex                       ( int dim, std::span<Pt> points );

    void            check_consistency             ( TF eps = 1e-6 ) const;
    void            for_each_vertex               ( auto &&func ) const;
    void            for_each_face                 ( auto &&func ) const;
    void            for_each_cut                  ( auto &&func ) const;
    void            display_vtk                   ( VtkOutput &vo ) const;
    TF              measure                       () const;

    PI              nb_vertices                   () const { return vertices.size(); }
    PI              dim                           () const { return pf.dim(); }

    void            cut                           ( const Pt &dir, TF sp, PI id );

    static void     _for_each_2_comb_excepted     ( PI size, PI excepted, auto&& func );
    void            _cut_int_ext_edge             ( PI n0, EdgeLink &e0, TF s0, TF s1 );
    void            _add_measure_rec              ( TF &res, SimpleSquareMatrix<TF,ct_dim,Arch> &M, const auto &cut_indices, PI prev_vertex_index ) const;

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

} // namespace sdot


template<class TF,int ct_dim,class Arch>
std::ostream& operator<<( std::ostream& os, const sdot::Cell<TF,ct_dim,Arch>& p );

#include "Cell.cxx"

#include "Cell_2D.h" // IWYU pragma: export
