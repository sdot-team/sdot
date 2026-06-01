from typing_extensions import Optional


class FfiCode:
    def __init__( self, fwd: str, bwd = "", header = "", includes : Optional[ list[ str ] ] = None, name = "" ) -> None:
        if includes is None:
            includes = []
        self.includes = includes
        self.header = header
        self.name = name
        self.fwd = fwd
        self.bwd = bwd

    def signature( self ) -> str:
        lst = [
            self.fwd,
            self.bwd,
            self.header
        ] + self.includes

        if self.name:
            lst = [ self.name ] + lst

        return " ".join( lst )

    @property
    def has_grad_code( self ):
        return bool( self.bwd )
