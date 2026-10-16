#pragma once

// =====================================================================================
// THE LINEAR ALGEBRA OF THE ADJOINT: what `d weights / d ( seeds, masses, density )` needs of the solver, and nothing else.
//
// At the fitted weights `w`, the residual `F( w, p, nu, rho ) = m( w, p, rho ) - nu` vanishes and its jacobian
// with respect to `w` is the laplacian of the Laguerre graph ( `Laplacian.h` ) -- the very matrix of the last Newton
// step. By the implicit function theorem
//
//     d w = - L^-1 ( d_p F dp + d_rho F drho - d nu )
//
// so the derivative of the solve is ONE solve with that matrix, on the diagram that measured the residual. `laplacian_solve`
// is that solve: `x = L^+ rhs`, the gauge `x_0 = 0` being the linear solver's responsibility. `L` is symmetric: the
// ADJOINT of the map `rhs -> x` is the same map, so the forward and the backward of the call run this one function.
//
// `pd` carries the weights ALREADY ( a diagram built at `w`, its majorants included ): nothing is written on it. With GROUPS, the
// solve is that of the ROOT level ( `Groups.h::aggregated` ): the inner levels are `SdotPlanNd`'s ( `groups.TreeSystem` ).
// =====================================================================================

#include "Linear.h"
#include "Sweep.h"

#include <cmath>
#include <limits>

namespace sdot {
namespace sdotplan {

/// `x = L^+ rhs` ( both in user order ), `L` the laplacian of the diagram `pd_in` for the density `dist`. `method`: a `Lin`.
/// A linear solver that fails leaves NaNs in `x`. `batch_axes`: the items of a `vmap` ( `jax.jacrev`, or a `vmap` of a `vjp`: a batch
/// of right-hand sides, the diagram being the same ) -- the diagram is swept and the matrix assembled ONCE, then each item is solved,
/// `rhs( batch_index )` into `x( batch_index )`.
template<class TK>
void laplacian_solve( const CpuQueue &queue, const auto &batch_axes, const auto &pd_in, const auto &dom, const auto &dist, const auto &rhs,
                      int method, SI cap0, auto &&x ) {
    auto pd = pd_in;                                     // a copy of the VIEWS: `Sweep` wants a writable diagram it will not write
    using PD = DECAYED_TYPE_OF( pd );
    using Dist = DECAYED_TYPE_OF( dist );
    constexpr int D = PD::ct_dim;
    const SI n = pd.nb_seeds();

    Sweep<PD,DECAYED_TYPE_OF( dom ),Dist,TK> bal( queue, pd, pd_in, dom, dist, cap0 );
    std::vector<double> a;
    std::vector<Facet> fa;
    bal.measures( a, &fa );

    // with a SINK ( partial transport ) the graph has one more node, 0, whose weight is the gauge ( `Sweep.h` ): the system
    // of the seeds is that of the graph with the sink's row taking the opposite of their sum, `x_0 = 0` -- `rhs` need not sum to zero
    const SI off = bal.off, nn = n + off;
    // GROUPS ( `Groups.h` ): the ROOT LEVEL -- the laplacian of the facets between roots, each root's members summed and linked to
    // it: `x` is the same for all the members of a root, from the sum of their right-hand sides ( `S^T L_R^+ S`, symmetric too )
    bool grouped = false;
    if constexpr ( decltype( bal )::groups ) grouped = bal.grp.active;
    std::vector<Facet> fa_root;
    if constexpr ( decltype( bal )::groups ) if ( grouped ) bal.grp.aggregated( fa, fa_root );
    Laplacian L;
    L.assemble( nn, grouped ? fa_root : fa );
    auto lin = linear_solver( Lin( method ), nn, D );
    lin->order( bal.rank_of );

    std::vector<double> b( nn ), d( nn );
    for ( SI item = 0; item < SI( batch_axes.size() ); ++item ) {
        const auto batch_index = batch_axes[ item ];
        auto &&r = rhs( batch_index );
        auto &&y = x( batch_index );
        std::fill( b.begin(), b.end(), 0.0 );
        std::fill( d.begin(), d.end(), 0.0 );
        for ( SI i = 0; i < n; ++i ) {
            b[ off + i ] = double( r( i ) );
            if ( off ) b[ 0 ] -= b[ off + i ];
        }
        if constexpr ( decltype( bal )::groups ) {
            if ( grouped ) {                             // ( the members' rows onto their root's )
                for ( SI i = 0; i < nn; ++i ) if ( bal.grp.root_of[ i ] != i ) { b[ bal.grp.root_of[ i ] ] += b[ i ]; b[ i ] = 0; }
            }
        }
        if ( ! lin->solves( L, b, d ) )
            std::fill( d.begin(), d.end(), std::numeric_limits<double>::quiet_NaN() );
        for ( SI i = 0; i < n; ++i ) {
            SI r = off + i;
            if constexpr ( decltype( bal )::groups ) if ( grouped ) r = bal.grp.root_of[ r ];
            y( i ) = d[ r ];
        }
    }
}

} // namespace sdotplan
} // namespace sdot
