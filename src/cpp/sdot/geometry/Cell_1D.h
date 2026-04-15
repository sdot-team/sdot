#pragma once

#include "Cell.h"

namespace sdot {

//
template<class TF,class Arch,class _CellInfo,class _CutInfo>
class Cell<TF,1,Arch,_CellInfo,_CutInfo> {
public:
    using               CellInfo = _CellInfo;
    using               CutInfo  = _CutInfo;

    static constexpr PI ct_dim   = 1;

    using               Pt       = DsVec<TF,ct_dim,Arch>;
    using               It       = DsVec<PI,ct_dim,Arch>;
    using               VI       = std::vector<int>;
    using               VF       = std::vector<TF>;

    struct Bound {
        // cut data
        Pt      cut_dir    = { Size(), 1 };
        TF      cut_dot;
        CutInfo info       = { 1 };

        // vertex data
        Pt      vertex_pos = { Size(), 1 };
        bool    vertex_ext;
    };

    using           Bounds                         = std::array<Bound,2>;

    /**/            Cell                          ( int dim = ct_dim );

    static Cell     axis_aligned_hypercube        ( Pt p0, Pt p1, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     axis_aligned_simplex          ( int dim, TF length, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     englobing_simplex             ( Pt center, TF radius, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     simplex                       ( int dim, std::span<Pt> points, CellInfo cell_info = {}, CutInfo cut_info = {} );

    friend std::ostream& operator<<( std::ostream &os, const Cell &p ) { for ( const auto &v : p.bounds ) os << "\n  pos: " << v.vertex_pos << " dir: " << v.cut_dir << " dot: " << v.cut_dot << " ext: " << v.vertex_ext; return os; }

    void            for_each_vertex               ( auto &&func ) const;
    //void          for_each_face                 ( auto &&func ) const;
    void            for_each_cut                  ( auto &&func ) const; ///< func( v.cut_dir, v.cut_dot, v.info )

    //void          check_consistency             ( TF eps = 1e-6 ) const;
    void            display_vtk                   ( VtkOutput &vo ) const;

    TF              for_each_cut_with_measure     ( auto &&f ) const; ///< f( cut_info, cut_measure, ext ). return measure
    auto            measure                       ( auto &&pos ) const; ///< pos( std::array<PI,3>( local_vertex_index_... ) )
    TF              measure                       () const;

    PI              nb_vertices                   () const { return 2; }
    PI              dim                           () const { return ct_dim; }

    void            cut                           ( const Pt &cut_dir, TF cut_dot, CutInfo cut_info );

    Bounds          bounds;                       ///<
    CellInfo        info;                         ///<
};

} // namespace sdot

#include "Cell_1D.cxx"
