"""The seeds as they came: `seed_positions [ n, d ]` ( `positions`, folded by a periodicity ), `weights [ n ]`, and no
acceleration -- each cell is cut by the `n - 1` bisectors, in order.

The floor against which `PowerDiagram_Bsp` is measured ( `O( n² )` ), and what remains when the
positions are a tracer, on which no tree can be built. Kernel side: `PowerDiagram_Plain.h`.
"""

import loom
from loom.tensor import RealTensor

from .PowerDiagram import PowerDiagram


class PowerDiagram_Plain( PowerDiagram ):
    seed_positions : RealTensor[ "num_point", "dim" ]

    # OPTIONAL, like the domain: absent ( `Unbound`, `NoneTensor` on the C++ side ), the weight term
    # vanishes from the plane AT COMPILE TIME and the diagram is the Euclidean one. "No weights" is a
    # STATE, not an array of zeros.
    weights   : RealTensor[ "num_point" ]

    def _init_seeds( self, positions, weights, accelerator ):
        return { "seed_positions": positions, **( {} if weights is None else { "weights": weights } ) }

    def _stored_positions( self ):
        return self.seed_positions

    @property
    def positions( self ):
        """`[ n, d ]`, as given ( see `PowerDiagram.positions` )"""
        return self._given_positions( self.seed_positions )

    @positions.setter
    def positions( self, positions ):
        if self.group_rep.is_defined:
            raise NotImplementedError( "positions of a diagram with groups: build a new `PowerDiagram` ( its groups are found for its positions )" )
        self.seed_positions = self._folded_positions( positions )
        self._dom_cell = None

    def _ranks_of_items( self ):
        import numpy as np
        return np.arange( int( self.nb_points.value ) )

    def _grad_seeds_expr( self ):
        return "grad_of_inputs.power_diagram.seed_positions, grad_of_inputs.power_diagram.weights"

    # ---- what the `SdotPlanNd` solver writes ( see `PowerDiagram_Bsp` ) ----------------------------

    def _solver_weights_call( self ):
        w = RealTensor[ self.num_point ]()
        return ( "inputs.power_diagram.with_weights( outputs.weights_out )",
                 dict( weights_out = loom.out( w ) ), dict( weights_out = w ) )

    def _solver_weights_after( self, produced ):
        self.weights = produced[ "weights_out" ].raw
