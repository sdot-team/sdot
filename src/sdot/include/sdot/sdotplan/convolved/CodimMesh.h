#pragma once

// `Convolved` of a codimension >= 1 mesh ( see `sdotplan/Convolved.h` ). Included at the end of `CodimMesh.h`.
#include <optional>
#include "../Convolved.h"

namespace sdot {
namespace sdotplan {

/// the same envelope, the width set to the stage: `rho exp( -d^2 / s^2 )` is the smoothing itself, nothing to compute. The
/// last stage ( `s = 0` ) is the density at its own width `params( 0 )`, which is also the smallest scale of the continuation
template<class... T>
struct Convolved<CodimMesh<T...>> {
    using Dist = CodimMesh<T...>;
    const Dist &dist;
    std::optional<Dist> current;
    static constexpr bool possible = true;

    Convolved( const Dist &dist ) : dist( dist ) {}

    const Dist &at( double s ) {
        if ( ! ( s > double( dist.params( 0 ) ) ) ) return dist;
        current.emplace( dist );
        current->conv_s = typename Dist::TF( s );
        current->whole = std::make_shared<std::vector<std::atomic<double>>>( std::size_t( dist.simplices.shape( 0 ) ) );   // ( a new width )
        for ( auto &v : *current->whole ) v.store( std::numeric_limits<double>::quiet_NaN(), std::memory_order_relaxed );
        return *current;
    }
    double min_scale( double ) const { return double( dist.params( 0 ) ); }
};

} // namespace sdotplan
} // namespace sdot
