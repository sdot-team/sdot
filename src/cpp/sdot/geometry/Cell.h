#pragma once

#include "RecursiveMapOfUniqueSortedIndices.h"
#include "../support/PointFactory.h"
#include "SimpleSquareMatrix.h"
#include "VtkOutput.h"

namespace sdot {


//
template<class TF,int ct_dim,class Arch>
struct StdCellInfo {
    using Pt  = Point<TF,ct_dim,Arch>;

    PI    global_dirac_index = PI( -1 );
    PI    local_dirac_index = PI( -1 );
    Pt    dirac_position;
    TF    dirac_weight;
    TF    potential;
};

//
template<class TF,int ct_dim,class Arch,class _CellInfo=StdCellInfo<TF,ct_dim,Arch>,class _CutInfo=StdCellInfo<TF,ct_dim,Arch>>
class Cell {
public:
    using       CellInfo = _CellInfo;
    using       CutInfo  = _CutInfo;

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
        TF   dot;
        PI   id;
        bool ext;
    };

    using           Vertices                      = std::vector<Vertex>;
    using           Cuts                          = std::vector<Cut>;

    /**/            Cell                          ( int dim );

    static Cell     axis_aligned_hypercube        ( Pt p0, Pt p1, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     axis_aligned_simplex          ( int dim, TF length, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     englobing_simplex             ( Pt center, TF radius, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     simplex                       ( int dim, std::span<Pt> points, CellInfo cell_info = {}, CutInfo cut_info = {} );

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

    friend std::ostream& operator<<( std::ostream& os, const Cell &p ) {
        os << "vertices:";
        for ( const auto& v : p.vertices ) {
            os << "\n  pos: " << v.pos << " cuts: " << v.cut_indices << " edge: ";
            for ( const auto& edge : v.edge_links ) {
                os << "[ n: " << edge.vertex_index << ", c: ";
                for ( sdot::PI i = 0, c = 0; i < v.cut_indices.size(); ++i )
                    if ( i != edge.num_cut_to_remove )
                        os << ( c++ ? ", " : "" ) << v.cut_indices[ i ];
                os << " ]";
            }
        }
        os << "\ncuts:";
        for ( const auto& v : p.cuts )
            os << "\n  dir: " << v.dir << " sp: " << v.sp << " id: " << v.id << " ext: " << v.ext;
        return os;
    }

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

    CellInfo        info;                         ///<
};

} // namespace sdot


template<class TF,int ct_dim,class Arch>
std::ostream& operator<<( std::ostream& os, const sdot::Cell<TF,ct_dim,Arch>& p );

#include "Cell.cxx"

#include "Cell_1D.h" // IWYU pragma: export
#include "Cell_2D.h" // IWYU pragma: export
