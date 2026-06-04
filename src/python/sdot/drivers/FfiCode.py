class FfiCode:
    """ """
    def __init__( self, fwd_code: str = "", bwd_code: str = "",
                  includes: "list | dict[ str, list ]" = None,
                  header: "str | dict[ str, str ]" = None,
                  name: str = "" ) -> None:
        self.fwd_code = fwd_code
        self.bwd_code = bwd_code
        self.includes = { "*": includes } if isinstance( includes, list ) else ( includes or {} )
        self.header   = { "*": header }   if isinstance( header  , str  ) else ( header   or {} )
        self.name     = name

    @staticmethod
    def parallel( batch_axes: list[ str ], fwd_code: str, bwd_code = "", name = "" ):
        return FfiCode(
            header = {
                "fwd": """
                    struct ParallerFwd {
                        HD void operator()( auto batch_index, auto &&p ) const {
                            FWD_CODE
                        }
                    };
                """.replace( "FWD_CODE", fwd_code ),
                "bwd": """
                    struct ParallerBwd {
                        HD void operator()( auto batch_index, auto &&p ) const {
                            BWD_CODE
                        }
                    };
                """.replace( "BWD_CODE", bwd_code ),
            },
            fwd_code = """
                run_parallel( cartesian_product_ranges( tuple( BATCH_SIZES ) ), ParallerFwd(), p );
            """.replace( "BATCH_SIZES", ", ".join( batch_axes ) ),
            bwd_code = """
                run_parallel( cartesian_product_ranges( tuple( BATCH_SIZES ) ), ParallerBwd(), p );
            """.replace( "BATCH_SIZES", ", ".join( batch_axes ) ) if bwd_code else "",
            name = name,
        )

    def header_for( self, pass_name: str ) -> str:
        return self.header.get( "*", "" ) + self.header.get( pass_name, "" )

    def includes_for( self, pass_name: str ) -> list[ str ]:
        return self.includes.get( "*", [] ) + self.includes.get( pass_name, [] )

    def signature( self ) -> str:
        parts = [ self.fwd_code, self.bwd_code, repr( self.header ), repr( self.includes ) ]
        if self.name:
            parts.insert( 0, self.name )
        return "__".join( parts )

    @property
    def has_grad_code( self ) -> bool:
        return bool( self.bwd_code )
