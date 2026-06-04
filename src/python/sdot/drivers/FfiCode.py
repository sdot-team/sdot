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

    @staticmethod
    def parallel( batch_axes: list[ str ], fwd: str, bwd = "", name = "" ):
        return FfiCode(
            header = """
                struct ParallerRun {
                    HD void operator()( auto batch_index, auto &&p ) const {
                        FWD_CODE
                    }
                };
            """.replace( "FWD_CODE", fwd ),
            fwd = """
                run_parallel( cartesian_product_ranges( tuple( BATCH_SIZES ) ), ParallerRun(), p );
            """.replace( "BATCH_SIZES", ", ".join( batch_axes ) ),
            name = name,
        )

    def signature( self ) -> str:
        lst = [
            self.fwd,
            self.bwd,
            self.header
        ] + self.includes

        if self.name:
            lst = [ self.name ] + lst

        return "__".join( lst )

    @property
    def has_grad_code( self ):
        return bool( self.bwd )
