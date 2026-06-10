from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AxisTerm:
    """
    One term of an axis-size expression: `coeff * <variable>`.

    The variable is referenced by `name`, with optional syntax attached:
        `arguments`  set by `( ... )` — the variable is expanded into one axis per value of
                     each argument (e.g. `shape( dim )` spans `dim` axes).
        `selection`  set by `[ ... ]` — the variable is a dynamic tensor indexed by these
                     expressions; it is then exposed under `max_of_<name>`.
    """

    coeff     : int
    name      : str
    arguments : list | None = None  # list[ AxisExpr ] — ( dim ): one axis per value of dim
    selection : list | None = None  # list[ AxisExpr ] — [ dim, nb_points ]: dynamic tensor index

    @property
    def exposed_name( self ) -> str:
        """Name under which the variable is exposed (dynamic selection -> max_of_*)."""
        if self.selection is not None:
            return "max_of_" + self.name
        return self.name

    def variable_value( self, system, forbidden_names ):
        """Resolve the variable (without `coeff`) through `system`, or None."""
        return system.value_of( self.exposed_name, forbidden_names )

    def unidimensional_version( self ) -> AxisTerm:
        arguments = None
        if self.arguments is not None:
            arguments = []
            for expr in self.arguments:
                nexpr = expr.unidimensional_version()
                if nexpr.always_one:
                    continue
                arguments.append( nexpr )
            if len( arguments ) == 0:
                arguments = None

        selection = None
        if self.selection is not None:
            selection = []
            for expr in self.selection:
                nexpr = expr.unidimensional_version()
                if nexpr.always_one:
                    continue
                selection.append( nexpr )

        return AxisTerm(
            coeff     = self.coeff,
            name      = self.name,
            arguments = arguments,
            selection = selection,
        )
