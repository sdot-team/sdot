from dataclasses import dataclass

@dataclass
class StructInfo:
    includes: set
    include : str
    params  : dict[ str ] # "TF": "class", "ct_dim": "int", ...
    name    : str #

    def add_param( self, name, kind ):
        if name not in self.params:
            self.params[ name ] = kind

