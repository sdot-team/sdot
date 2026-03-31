#pragma once

#include "Cell.h"

namespace sdot {

//
template<class TF,class Arch>
class Cell<TF,2,Arch> {
public:
    static constexpr PI ct_dim   = 2;
    using               Pt       = Point<TF,ct_dim,Arch>;
    using               It       = Point<PI,ct_dim,Arch>;
    using               VI       = std::vector<int>;
    using               VF       = std::vector<TF>;

    struct VertexAndCut {
        bool outside_the_cut() const { return sp > 0; }
        bool inside_the_cut() const { return ! ( sp > 0 ); }

        // cut data
        Pt   cut_dir;
        TF   cut_dot;
        PI   cut_id;

        // vertex data
        Pt   pos;
        TF   sp;
        bool ext;
    };

    using           Vertices                      = std::vector<VertexAndCut>;

    /**/            Cell                          ( int dim = ct_dim );

    static Cell     axis_aligned_hypercube        ( Pt p0, Pt p1, bool ext = 0 );
    static Cell     axis_aligned_simplex          ( int dim, TF length );
    static Cell     englobing_simplex             ( Pt center, TF radius );
    static Cell     simplex                       ( int dim, std::span<Pt> points );

    void            for_each_vertex               ( auto &&func ) const;
    void            for_each_face                 ( auto &&func ) const;
    void            for_each_cut                  ( auto &&func ) const;

    void            check_consistency             ( TF eps = 1e-6 ) const;
    void            display_vtk                   ( VtkOutput &vo ) const;
    TF              measure                       () const;

    PI              nb_vertices                   () const { return vertices.size(); }
    PI              dim                           () const { return ct_dim; }

    void            cut                           ( const Pt &cut_dir, TF cut_dot, PI cut_id );

    Vertices        vertices;                     ///<
};

} // namespace sdot

template<class TF,class Arch>
std::ostream& operator<<( std::ostream& os, const sdot::Cell<TF,2,Arch> &p );

#include "Cell_2D.cxx"
