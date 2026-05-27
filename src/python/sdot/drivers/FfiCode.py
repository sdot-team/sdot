from typing_extensions import Optional


class FfiCode:
    def __init__( self, fwd: str, bwd = "", header = "", includes : Optional[ list[ str ] ] = None ) -> None:
        if includes is None:
            includes = []
        self.includes = includes
        self.header = header
        self.fwd = fwd
        self.bwd = bwd

    def signature( self ) -> str:
        return " ".join( [
            self.fwd,
            self.bwd,
            self.header
        ] + self.includes )

    @property
    def has_grad_code( self ):
        return bool( self.bwd )
