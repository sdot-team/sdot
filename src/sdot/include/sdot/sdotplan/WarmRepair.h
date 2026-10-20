#pragma once

// THE REPAIR OF A WARM START, the same for the CPU solver ( `Solve.h` ) and the card's ( `gpu/Newton2D.cuh` ).
//
// A warm start ( the weights of a neighbouring fit ) that empties a few cells is not thrown away for the Voronoi: the cells it
// empties are those of NEAR-COINCIDENT PAIRS ( two seeds `d << ` the spacing apart -- in a projection, two points far apart in
// depth ), whose dividing plane moves by `dw / 2 d`: a move far smaller than the spacing swaps the pair's split, one cell holds
// both masses and the other none. Each such cell searches its OWN weight, alone, one diagram per round: up by `step`, `step`
// times 4, from a tiny step ( `STEP0` of the squared spacing ) until it holds `LO` of its target, then a BISECTION between the
// last weight too small and the first too big ( over `HI` of its target: it does not starve its partner ). Steps sized on the
// spacing itself ( `dw ~ l^2` ) are `l / d` too large for a pair: each revived cell then empties its partner, and the
// revivals cascade -- the reason why "reviving the empty cells" had been rejected.
//
// A cell whose bracket closes without passing ( its mass jumps from too small to too big ) keeps the high end. The repair stops
// after `MAX_ROUNDS` diagrams, and the caller keeps it only if its smallest mass beats the warm start's ( the Voronoi stays the
// last resort ). It is only tried when few cells need it ( `MAX_FRACTION` of the seeds ): a move that swaps 5 % of the cells
// ( 1e5 diracs moved by a third of their spacing ) is repaired to an admissible start from which Newton nevertheless fails.

#include <cmath>

#if defined( __CUDACC__ )
#define SDOT_WR_HD __host__ __device__ __forceinline__
#else
#define SDOT_WR_HD inline
#endif

namespace sdot {
namespace sdotplan {

struct WarmRepair {
    static constexpr double LO = 0.1;                    ///< a cell under `LO nu_i` is repaired
    static constexpr double HI = 1.75;                   ///< ... and does not take more than `HI nu_i`
    static constexpr double STEP0 = 1e-4;                ///< the first step, in squared spacings
    static constexpr double CLOSED = 1e-6;               ///< a bracket narrower than this ( squared spacings ) is closed
    static constexpr int MAX_ROUNDS = 24;                ///< diagrams at most
    static constexpr double MAX_FRACTION = 1e-2;         ///< tried only when at most this fraction of the seeds needs it ( + 8 )

    /// whether a repair of `count` cells among `n` is worth trying
    static bool worth( double count, double n ) { return count <= MAX_FRACTION * n + 8; }
};

/// the start of the repair for one seed: no bracket ( `lo`, `hi`: -/+ HUGE_VAL ), the first step, not touched ( `touched < 0`: a seed
/// left out, an exact duplicate whose cell is empty by construction )
SDOT_WR_HD void warm_repair_init( double &lo, double &hi, double &step, double &touched, double l2, bool left_out ) {
    lo = -HUGE_VAL;
    hi = HUGE_VAL;
    step = WarmRepair::STEP0 * l2;
    touched = left_out ? -1 : 0;
}

/// one round for one seed, from `a` the mass of its cell at `w`: `w` moved if the seed is still being repaired ( the return )
SDOT_WR_HD bool warm_repair_cell( double a, double nu, double &w, double &lo, double &hi, double &step, double &touched, double l2 ) {
    if ( touched < 0 )
        return false;
    const bool small = a < WarmRepair::LO * nu, big = touched > 0 && a > WarmRepair::HI * nu;
    if ( hi - lo <= WarmRepair::CLOSED * l2 ) {       // a closed bracket: the high end, whatever its mass, and done
        touched = -1;
        if ( small ) w = hi;
        return small;
    }
    if ( ! small && ! big )
        return false;
    touched = 1;
    if ( small ) lo = w; else hi = w;
    if ( hi == HUGE_VAL ) { w += step; step *= 4; }            // ( no weight too big yet: up, faster and faster )
    else w = 0.5 * ( lo + hi );
    return true;
}

} // namespace sdotplan
} // namespace sdot
