"""RADIAL FUNCTIONS AROUND THE DIRACS: what a cell integrates beyond the density, a function of the power distance to its seed.

`PowerDiagram( ..., radial_func = RadialExp( eps ) )`: the cell of seed `i` ( its Laguerre cell, unchanged ) integrates

    rho( x ) exp( ( w_i - |x - p_i|² ) / ( 2 eps ) )

instead of `rho( x )`. It is the measure of the Moreau-Yosida regularization of the entropy, `min_sigma W_2²( sigma, mu ) / ( 2 eps )
+ int sigma log sigma` ( Leclerc, Mérigot, Santambrogio, Stra, 2020 ): when the weights give each cell the mass of its dirac, that
integrand IS the optimal `sigma` ( see `docs/examples/fokker-planck.md` ). `OtProblem( ..., radial_func = ... )` solves for
those weights. The C++ side is `integration/RadialExp.h`.
"""


class RadialExp:
    """`exp( ( w_i - |x - p_i|² ) / ( 2 eps ) )` around each seed ( see the module ): a gaussian of variance `eps` centered on the seed,
    whose height grows with its weight."""

    def __init__( self, eps ):
        self.eps = float( eps )
        if not self.eps > 0:
            raise ValueError( f"RadialExp: `eps` has to be positive ( got { eps } )" )

    def __repr__( self ):
        return f"RadialExp( { self.eps } )"
