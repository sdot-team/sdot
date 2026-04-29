from dataclasses import dataclass

@dataclass
class StructInfo:
    includes: set
    include : str
    params  : dict[ str ] # "TF": "class", "ct_dim": "int", ...
    name    : str #

    avoid_params_in_inst: []

    def add_param( self, name, kind ):
        if name not in self.params:
            self.params[ name ] = kind

