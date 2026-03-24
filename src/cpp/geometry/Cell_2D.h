#pragma once

#include "Cell.h"

namespace sdot {

//
template< class TF>
class Cell<TF,2> {
public:
    static constexpr PI ct_dim   = 2;
    using               VF       = std::vector<TF>;
    using               VI       = std::vector<int>;
    using               Pt       = Point<TF,ct_dim>;
    using               It       = Point<PI,ct_dim>;

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

    void            for_each_cut                  ( auto &&func );

    void            check_consistency             ( TF eps = 1e-6 ) const;
    void            display_vtk                   ( VtkOutput &vo ) const;
    TF              measure                       () const;
    PI              dim                           () const { return ct_dim; }

    void            cut                           ( const Pt &cut_dir, TF cut_dot, PI cut_id );

    Vertices        vertices;                     ///<
};

} // namespace sdot

T_T std::ostream& operator<<( std::ostream& os, const sdot::Cell<T,2> &p );

#include "Cell_2D.cxx"
