#pragma once

#include "support/DsVecFactory.h"
#include "support/TensorView.h"
#include "geometry/Cell.h"

namespace sdot {

///
///
///
template<class TF,int ct_dim,int order,class Arch>
struct PolynomialGrid {
    static constexpr PI nb_coeffs             = pow_rec( order + 1, ct_dim ); ///< Q_k: (order+1)^dim
    using               PiFactory             = DsVecFactory<PI,ct_dim,Arch>;
    using               PfFactory             = DsVecFactory<TF,ct_dim,Arch>;
    using               Values                = TensorView<const TF,ct_dim+1,Arch>; ///< [ x, y, z, num_poly ]
    using               Bounds                = TensorView<const TF,2,Arch>;
    using               Knots                 = std::vector<TensorView<const TF,1,Arch>>;
    using               Pp                    = DsVec<PI,ct_dim+1,Arch>;
    using               Pi                    = DsVec<PI,ct_dim,Arch>;
    using               Pt                    = DsVec<TF,ct_dim,Arch>;

    struct              Polynomial            { Polynomial() : coeffs( Size(), nb_coeffs ) {} DsVec<TF,nb_coeffs,Arch> coeffs; /** Q_k basis, lex multi-index (p0,p1,...) each in 0..order. Ex order=1 dim=2: 1 y x xy */ };

    /**/                PolynomialGrid        ( Values values, Bounds bounds, const Knots &knots, bool normalize );

    bool                has_skew_or_rotation  () const { return false; }
    T_U auto            englobing_cell        ( PI dim, typename Cell<U,ct_dim,Arch>::CellInfo cell_info = {}, typename Cell<U,ct_dim,Arch>::CutInfo cut_info = {} ) const -> Cell<U,ct_dim,Arch>;
    void                for_each_piece        ( const auto &cell, auto &&func ) const; ///< func( new_cell, Polynomial )
    Polynomial          polynomial            ( Pi position ) const;
    auto                grid_shape            () const;
    TF                  knot                  ( PI d, PI index ) const;
    PI                  dim                   () const { return values.rank() - 1; }

    static auto         simplex_facet_integral( std::span<const Pt> vertices, const Polynomial &value ); ///<
    static auto         simplex_integral      ( std::span<const Pt> points, const Polynomial &value ); ///<
    static auto         facet_integral        ( auto facet, const Polynomial &value ); ///<
    static auto         integral              ( auto cell, const Polynomial &value ); ///< cell is expected

    TF                  piece_integral        ( Pi position, const Polynomial &value ) const; ///< integral on a piece of this (an item of the grid)

    // struct              Cursor                { TF mass, x0, x1, y0, y1; PI i1; }; ///< State for traversing the CDF;

    // /// Sub-interval [x0,x1] of rho, contained within grid cell [k0,k1]
    // struct Piece {
    //     TF  w2_dist( TF ref_x ) const;
    //     TF  moment () const;

    //     PI  i1;
    //     TF  k0, k1;
    //     TF  x0, x1;
    //     TF  y0, y1;
    // };

    // // 1D specific
    // Cursor       first_cursor                  () const;
    // Cursor       last_cursor                   () const;
    // PI           nb_values                     () const;
    // TF           min_x                         () const;
    // TF           max_x                         () const;
    // TF           knot                          ( PI i ) const;

    // void         apply_normalization_correction( TF g_dist, TF phi_avg, TensorView<TF,1,Cpu> grad_values ) const;
    // void         accumulate_gradients_dist     ( const Piece &p, TF g_dist, TF dirac_x, TF potential, TensorView<TF,1,Cpu> grad_values ) const;
    // void         take_some_mass                ( Cursor &, TF mass_to_take, auto &&func ) const;


private:
    void         _for_each_piece               ( const auto &cell, const auto &func, auto beg, auto end, auto &cur, PI d ) const;

    TF           coeff_values;
    bool         normalize;
    Values       values;
    Bounds       bounds;
    Knots        knots;
};

//
template<int order,class T,int v_rank,class Arch> auto polynomial_grid( CtInt<order>, TensorView<const T,v_rank,Arch> values, TensorView<const T,2,Arch> bounds, const std::vector<TensorView<const T,1,Arch>> &knots, bool normalize ) {
    return PolynomialGrid<T,v_rank-1,order,Arch>{ values, bounds, knots, normalize };
}

} // namespace sdot

#include "PolynomialGrid.cxx"
