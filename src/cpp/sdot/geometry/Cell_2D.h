#pragma once

#include <functional>
#include "Cell.h"

namespace sdot {

//
template<class TF,class Arch,class _CellInfo,class _CutInfo>
class Cell<TF,2,Arch,_CellInfo,_CutInfo> {
public:
    using               CellInfo = _CellInfo;
    using               CutInfo  = _CutInfo;

    static constexpr PI ct_dim   = 2;

    using               Pt       = Point<TF,ct_dim,Arch>;
    using               It       = Point<PI,ct_dim,Arch>;
    using               VI       = std::vector<int>;
    using               VF       = std::vector<TF>;

    struct Edge {
        bool outside_the_cut() const { return vertex_dot > 0; }
        bool inside_the_cut() const { return ! ( vertex_dot > 0 ); }

        // cut data
        Pt      cut_dir;
        TF      cut_dot;
        CutInfo info;

        // vertex data
        Pt      vertex_pos;
        TF      vertex_dot;
        bool    vertex_ext;
    };

    using           Edges                        = std::vector<Edge>;

    /**/            Cell                          ( int dim = ct_dim );

    static Cell     axis_aligned_hypercube        ( Pt p0, Pt p1, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     axis_aligned_simplex          ( int dim, TF length, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     englobing_simplex             ( Pt center, TF radius, CellInfo cell_info = {}, CutInfo cut_info = {} );
    static Cell     simplex                       ( int dim, std::span<Pt> points, CellInfo cell_info = {}, CutInfo cut_info = {} );

    void            for_each_vertex               ( auto &&func ) const;
    void            for_each_face                 ( auto &&func ) const;
    void            for_each_cut                  ( auto &&func ) const;

    void            check_consistency             ( TF eps = 1e-6 ) const;
    void            display_vtk                   ( VtkOutput &vo ) const;

    TF              for_each_cut_with_measure     ( auto &&f ) const; ///< f( cut_info, cut_measure, ext ). return measure
    TF              measure                       () const;

    PI              nb_vertices                   () const { return edges.size(); }
    PI              dim                           () const { return ct_dim; }

    void            cut                           ( const Pt &cut_dir, TF cut_dot, CutInfo cut_info );

    Edges           edges;                        ///<
    CellInfo        info;                         ///<
};

} // namespace sdot

template<class TF,class Arch>
std::ostream& operator<<( std::ostream& os, const sdot::Cell<TF,2,Arch> &p );

#include "Cell_2D.cxx"
