#pragma once

// the members + the axes this body names, written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/SumOfGaussians.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include "PointwiseDensity.h"
#include "GaussianTree.h"

namespace sdot {

// A sum of ISOTROPIC gaussians, seen as a density to integrate over a cell.
//
//   rho( x ) = somme_i  w_i * exp( - r_i^2 / ( 2 s_i^2 ) ) / ( 2 pi s_i^2 ) ^ ( d / 2 ),
//   r_i = | x - c_i |
//
// `w_i` is the MASS of gaussian `i`, not its height -- the total mass is therefore the sum of the
// weights, and normalizing is a division (see `distributions/SumOfGaussians.py`).
//
// It is the first density that is not piecewise constant, and all it has to say fits in three
// POINTWISE answers: the value, the gradient, and where to put `d rho / d parameters`.
// It CUTS nothing (its piece is the whole cell) and knows nothing about cells; it is
// `PowerDiagram::integrate_into` that, seeing `is_constant == false`, cuts them into simplices and
// does its quadrature there. The split is there, and it has no third side.
SDOT_TEMPLATE_DECL_FOR_SumOfGaussians
struct SumOfGaussians {
    SDOT_ATTRIBUTES_OF_SumOfGaussians

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( positions )::TF;

    /// THE CONVOLUTION by a gaussian of width `conv_s` ( `with_convolution` ): on a sum of
    /// gaussians it only changes the widths, `sigma_i' = sqrt( sigma_i^2 + conv_s^2 )`, nothing
    /// else -- what the width continuation of `SdotPlanNd` walks through ( `sdotplan/Continuation.h` ).
    /// A member APART from the generated attributes: `0` by default, hence absent from any ordinary call.
    TF conv_s = 0;

    /// THE LOCAL TERMS ( `GaussianTree.h` ). With a `tree` ( hung by `Convolved<SumOfGaussians>`, the solver ), each cell
    /// sees only the terms that weigh on it -- the gaussians near it, or groups of them merged at the continuation
    /// width -- and the sums below run over `terms` instead of the generated attributes. FORWARD ONLY: the adjoint
    /// ( `add_value_grad_at`, `integrate_over_simplex_bwd` ) always runs over the gaussians themselves, and a density
    /// with a tree is never differentiated ( the solver does not ).
    using Term = GaussianTerm<TF,ct_dim>;
    const GaussianTree<TF,ct_dim> *tree = nullptr;
    const Term *terms = nullptr;
    SI nb_local = 0;

    HD SI nb_terms () const { return terms ? nb_local : SI( nb_gaussians ); }
    HD TF center_of( SI i, PI c ) const { return terms ? terms[ i ].c[ c ] : TF( positions( i, c ) ); }
    HD TF weight_of( SI i ) const { return terms ? terms[ i ].w : TF( weights( i ) ); }

    HD TF sigma_of( SI i ) const {
        if ( terms ) return terms[ i ].s;                 // the convolution is already in
        const TF s = TF( sigmas( i ) );
        return conv_s > 0 ? sdot::sqrt( s * s + conv_s * conv_s ) : s;
    }
    HD SumOfGaussians with_convolution( TF s ) const { SumOfGaussians r( *this ); r.conv_s = s; return r; }
    /// the smallest width ( before convolution ): the scale below which the continuation stops
    HD TF smallest_sigma() const { TF r = TF( sigmas( 0 ) ); for ( SI i = 1; i < SI( sigmas.shape( 0 ) ); ++i ) r = sdot::fmin( r, TF( sigmas( i ) ) ); return r; }

    /// A single piece, the cell itself, and not a cut: nothing to cut when the density
    /// is defined everywhere by the same formula. The cutting scratch is therefore not touched (and
    /// `extra_cuts_per_piece` returns 0, so it is not even allocated).
    ///
    /// HOW we integrate depends on the dimension, and it is a choice made here, by
    /// composition -- the integrator only sees the contract:
    ///   * in 2D and 3D we KNOW how (see `wedge_measure`, `face3`), so we pass ourselves;
    ///   * beyond that we do not, so we declare ourselves a BLACK BOX by wrapping in
    ///     `PointwiseDensity`, which only needs `value_at` / `gradient_at`.
    ///
    /// With a `tree`, the piece is integrated by a copy of `self` that carries the terms of this cell.
    HD void for_each_piece( const auto &cell, auto &&/*ws*/, auto &&func ) const {
#if ! defined( __CUDA_ARCH__ )
        if ( tree && ! terms ) {
            double lo[ ct_dim ], hi[ ct_dim ];
            for ( int d = 0; d < ct_dim; ++d ) { lo[ d ] = 1e300; hi[ d ] = -1e300; }
            for ( SI v = 0; v < SI( cell.nb_vertices() ); ++v )
                for ( int d = 0; d < ct_dim; ++d ) {
                    const double x = double( cell.coord( int( v ), d ) );
                    lo[ d ] = x < lo[ d ] ? x : lo[ d ];
                    hi[ d ] = x > hi[ d ] ? x : hi[ d ];
                }
            thread_local std::vector<Term> buf;
            tree->gather( lo, hi, double( conv_s ), buf );
            SumOfGaussians local( *this );
            local.terms = buf.data();
            local.nb_local = SI( buf.size() );
            if constexpr ( ct_dim == 2 || ct_dim == 3 )
                func( cell, local );
            else
                func( cell, PointwiseDensity{ local } );
            return;
        }
#endif
        if constexpr ( ct_dim == 2 || ct_dim == 3 )
            func( cell, *this );
        else
            func( cell, PointwiseDensity{ *this } );
    }

    /// a term WIDE against the triangle `pts` ( during a continuation ) is integrated by a 7 point rule instead of the
    /// exact reduction: the error goes as `( diameter / width )^6`, and it costs 7 exponentials instead of ~100
    static constexpr TF quadrature_ratio = 1.5;
    HD TF quadrature_term( SI i, const auto &pts ) const;

    // ---- EXACT integration in 2D ------------------------------------------------------------------
    // An isotropic gaussian over a triangle has no elementary closed form -- it is Owen's T
    // function -- but it REDUCES to a smooth, bounded 1D integral, whose accuracy no longer depends
    // on the shape of the cell. That is the whole difference with a quadrature rule on the
    // triangle, whose error goes as `( cell size / sigma ) ^ 4`.
    //
    // The reduction: after translation/scaling, the triangle is decomposed into three signed
    // corners `( 0, P, Q )`, and a corner is integrated in polar coordinates --
    //
    //     W( P, Q ) = sign( p ) / 2pi * Int_{t_P}^{t_Q} [ 1 - exp( -( p^2 + t^2 ) / 2 ) ] p dt / ( p^2 + t^2 )
    //
    // where `p` is the signed distance from the origin to the line `PQ` and `t` the abscissa along it.
    // Written THIS WAY -- the difference taken under the integral, not between two arctangents -- it has
    // no catastrophic cancellation, even when the origin grazes the line.
    //
    // Beyond `|t| = tail_cut`, the exponential is worth nothing and only the
    // Lorentzian remains, whose antiderivative is `arctan( t / p )`: the tails are therefore EXACT and
    // free, and the quadrature only works on a bounded interval where the integrand has a
    // scale `>= 1`. Measured: `8 x 4 = 32` evaluations per edge give 4e-14 absolute error
    // on configurations chosen to be nasty (seed on the center, huge triangle, grazing
    // sliver).
    static constexpr bool is_constant = false;

    /// the moments ( `diagram::integrate_moments_into` ): in 2D and 3D the exact ones ( `integrate_moment_over_simplex`, the
    /// same as those the derivatives of a plan differentiate ), elsewhere the adaptive quadrature of `PointwiseDensity`.
    HD void integrate_moments_over_simplex( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
        if constexpr ( ct_dim == 2 ) {
            m += integrate_moment_over_simplex<3>( pts );
            mx[ 0 ] += integrate_moment_over_simplex<0>( pts );
            mx[ 1 ] += integrate_moment_over_simplex<1>( pts );
            m2 += integrate_moment_over_simplex<2>( pts );
        } else if constexpr ( ct_dim == 3 )
            moments3( pts, m, mx, m2 );                  // ( one pass over the faces for the three )
        else
            PointwiseDensity{ *this }.integrate_moments_over_simplex( pts, m, mx, m2 );
    }

    /// `Int_T p_K rho` over the triangle `pts`, EXACT, in 2D -- what `diagram::moment_dist<K>` integrates ( `K < 2`: `x_K`,
    /// `K == 2`: `|x|^2`, `K == 3`: `1`; in 3D `K < 3`, `3`, `4`, see `moment3` ). The adaptive quadrature it replaces could miss a gaussian by 10 % on a large
    /// cell, depending on how the cell was cut into triangles. In the standard frame of gaussian `i`, `y = ( x - c ) / s`,
    /// with `phi` the standard normal: `grad phi = - y phi` and `laplacian phi = ( |y|^2 - 2 ) phi`, so by the divergence
    /// theorem
    ///
    ///     Int_T y phi       = - sum_e n_e J_e               ( n_e the outward normal of edge `e`, J_e = Int_e phi ds )
    ///     Int_T |y|^2 phi   = 2 M - sum_e p_e J_e           ( p_e = y . n_e, constant along the edge, M = Int_T phi )
    ///
    /// -- the edge integrals are the `erf`s of `edge_info`, and `M` the exact reduction.
    template<int K> HD TF integrate_moment_over_simplex( const auto &pts ) const requires ( ct_dim == 2 || ct_dim == 3 );

    /// its adjoint, exact too ( `diagram/Moments.h` reads `exact_moments` ). Moving a vertex sweeps its two edges
    /// ( Reynolds ): `d I / d P_k = sum_{e ∋ k} n_e Int_e p rho lambda_k ds`, edge integrals of a polynomial times a
    /// gaussian, `erf`s and `exp`s ( `edge_moments` ). The parameters follow from invariances, with no other integral:
    ///   * translating the triangle and the center together: `d I / d c = Int_T grad p rho - sum_k d I / d P_k`;
    ///   * scaling them about the center: `s d I / d s = Int_T grad p . ( x - c ) rho - sum_k d I / d P_k . ( P_k - c )`;
    ///   * the weight: `I / w`.
    static constexpr bool exact_moments = ct_dim == 2 || ct_dim == 3;
    template<int K> HD void integrate_moment_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const requires ( ct_dim == 2 || ct_dim == 3 );

    // ---- EXACT integration in 3D ------------------------------------------------------------------
    // A tetrahedron goes to its four faces, in the standard frame `y = ( x - c ) / s` of each gaussian:
    //
    //   * the MASS is the sum of the signed cones from the center over the faces ( outward normals ): a face at the distance
    //     `h` gives `h / 4pi Int_face Phi3( r ) / r^3 dA` ( `Phi3` the law of `|X|` in R^3 ), whose radial integral is closed --
    //     what is left is one smooth integral along each edge, like the 2D corner ( `detail::g3_cone_edge` );
    //   * on a face, `phi = phi_1( h ) phi_2( u, v )` ( `y = h n + u e1 + v e2` ): its integrals are 2D moments of the face, from
    //     its corners ( `wedge_measure` ) and, by `Int u q phi_2 = Int d_u q phi_2 - sum_e n_u Int_e q phi_2`, from `erf`s along its
    //     edges ( `face3`, up to degree 3 );
    //   * the MOMENTS by divergence: `Int y phi = - sum_f n_f Int_f phi`, `Int |y|^2 phi = 3 M - sum_f h_f Int_f phi`;
    //   * the ADJOINT by Reynolds: moving vertex `k` sweeps its three faces, `sum_f n_f Int_f p phi lambda_k dA`; the center and the
    //     width from the invariances, as in 2D.
    //
    // A gaussian farther than `9 s` from the tetrahedron is nothing ( `far3` ).
    struct Face3 { TF n[ 3 ]; TF e1[ 3 ]; TF e2[ 3 ]; TF h; TF phi1; TF cone; TF M[ 10 ]; TF lam[ 3 ][ 3 ]; int vid[ 3 ]; };
    HD void face3( const TF ( &Y )[ 4 ][ 3 ], int opp, int deg, Face3 &f ) const;
    HD void tet3( SI i, const auto &pts, int deg, TF ( &Y )[ 4 ][ 3 ], Face3 ( &F )[ 4 ], TF &Mt, TF ( &Nv )[ 3 ], TF &Y2 ) const;
    HD bool far3( SI i, const auto &pts ) const;
    template<int K> HD TF   moment3    ( const auto &pts ) const;
    HD void                 moments3   ( const auto &pts, TF &m, auto &mx, TF &m2 ) const;
    template<int K> HD void moment3_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const;

    /// For an edge `A -> B` of the triangle `A, B, C`, in the standard frame: its OUTWARD normal and `y . n`, then
    /// `Int_e q phi ds` ( `j` ) and `Int_e q phi lambda_A ds` ( `ja` ) for `q = 1, y_0, y_1, |y|^2`.
    struct EdgeMoments { Vector<TF,2> n; TF p; TF j[ 4 ]; TF ja[ 4 ]; };
    HD EdgeMoments edge_moments( const auto &A, const auto &B, const auto &C ) const;

    static constexpr TF  tail_cut   = 8;    ///< `exp( -t^2/2 ) < 1e-14` beyond: the tail is exact
    static constexpr int nb_panels  = 4;    ///< Gauss-Legendre panels over the core

    /// The SIGNED standard normal measure of the triangle `( 0, P, Q )` -- the corner. `P` and `Q` are 2D whatever
    /// `ct_dim`: the in-plane frame of a 3D facet ( `facet_mass` ) goes through here too.
    HD TF wedge_measure( const auto &P, const auto &Q ) const;

    /// The standard normal measure of the triangle `ys` (positive, any orientation).
    HD TF std_triangle_measure( const auto &ys ) const;

    /// `Int_T rho`, exact. `pts`: the 3 vertices ( 4 in 3D ).
    HD TF integrate_over_simplex( const auto &pts ) const;

    /// The ELEMENTARY adjoint -- that is the remarkable point: the value requires a special
    /// function, its derivatives do not. Everything reduces to BOUNDARY integrals, which for a gaussian
    /// along a segment are `erf`s:
    ///   * a vertex: moving it sweeps its two edges, hence `Int rho * lambda` on each;
    ///   * the center: `d/dc = -Int grad rho = -Contour rho n ds` (divergence) -- equal, and it is a
    ///     good check, to MINUS the sum of the per-vertex derivatives;
    ///   * sigma: `d rho / d sigma = sigma * laplacian( rho )` (heat identity, `t = sigma^2/2`),
    ///     hence again a boundary flux, and `y . n` is CONSTANT there along an edge;
    ///   * the weight: the measure itself, already computed.
    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const;

    /// For an edge `A -> B` of the triangle `A, B, C`, in the standard frame: the OUTWARD normal, the
    /// signed distance `y . n` (constant along the edge), `Int phi ds`, and `Int phi lambda_A ds`.
    struct EdgeInfo { Vector<TF,2> n; TF p; TF j0; TF j1a; };
    HD EdgeInfo edge_info( const auto &A, const auto &B, const auto &C ) const;

    /// The NORMALIZED kernel of gaussian `i` at `x` (mass 1), and the squared distance --
    /// the two quantities from which everything else follows, computed once.
    HD auto kernel_at( SI i, const auto &x ) const;

    /// `Int_{facet} rho ds` over the facet `cut` of the cell `pc` -- what the laplacian of a transport reads
    /// ( `sdotplan/Sweep.h` ). In 2D, the edge `[ v_cut, v_cut+1 ]`: a gaussian along a segment is an `erf`, the
    /// distance to the segment being constant. In 3D, a planar polygon: an isotropic gaussian SPLITS into a 1D factor
    /// in the distance to the plane and a 2D gaussian in the plane, whose measure over the polygon is the exact 2D
    /// reduction ( `wedge_measure` ) on a fan of triangles. The proxy of a 3D mesh needs it ( `SdotPlanNd._solve_by_proxy` ).
    HD TF   facet_mass    ( const auto &pc, int cut ) const requires ( ct_dim == 2 || ct_dim == 3 );
    HD TF   facet_mass_2d ( const auto &pc, int cut ) const;
    HD TF   facet_mass_3d ( const auto &pc, int cut ) const;   ///< `pc.for_each_facet_triangle` ( `LocalN` )

    HD TF   value_at      ( const auto &x ) const;   ///< rho( x )
    HD auto gradient_at   ( const auto &x ) const;   ///< grad rho( x ), a `Vector<TF,ct_dim>`

    /// Accumulates `g * d rho( x ) / d parameter` into the cotangent of each parameter.
    ///
    /// It is the "distribution" half of the adjoint, and it only knows `x`: the integrator
    /// passes it the quadrature weight of the node (`g`), without knowing what is behind it. The
    /// additions are ATOMIC -- a wide gaussian is seen by the work-items of many
    /// cells at once -- and each is guarded by the validity of its target, a non-differentiated
    /// parameter arriving as a `NoneTensor`.
    HD void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const;

    /// what this density's adjoint derives, for loom's `computed_grads` ( `flags`: the distribution's, under one
    /// output ). The masses are never read by the integration: their derivative is known to be zero.
    static void computed_grads( auto &&flags ) {
        flags.positions = flags.sigmas = flags.weights = true;   // all three, `add_value_grad_at` / `integrate_over_simplex_bwd`
        if constexpr ( requires { flags.current_mass; } ) flags.current_mass = true;
        if constexpr ( requires { flags.target_mass; } )  flags.target_mass  = true;
    }
};

}

#include "SumOfGaussians.cxx"

// its convolved density, for the transport solver ( `sdotplan/Convolved.h` )
#include "sdotplan/convolved/SumOfGaussians.h"
