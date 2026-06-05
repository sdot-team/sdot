class FfiCode:
    """ Abstract base for all FfiCode variants. Subclasses generate C++ on demand. """

    def with_prepended_batch_axis( self, name: str ) -> 'FfiCode':
        raise NotImplementedError( "vmap requires FfiCodeParallel or a custom with_prepended_batch_axis implementation" )

    def header_for( self, pass_name: str ) -> str:
        raise NotImplementedError

    def includes_for( self, pass_name: str ) -> list[ str ]:
        raise NotImplementedError

    def signature( self ) -> str:
        raise NotImplementedError

    @property
    def has_grad_code( self ) -> bool:
        raise NotImplementedError

    def code_for( self, pass_name: str ) -> str:
        raise NotImplementedError

    @property
    def name( self ) -> str:
        raise NotImplementedError


class FfiCodeCustom( FfiCode ):
    """ FfiCode with manually written C++ strings — no automatic vmap support. """

    def __init__( self, fwd_code: str = "", bwd_code: str = "",
                  includes: "list | dict[ str, list ]" = None,
                  header: "str | dict[ str, str ]" = None,
                  name: str = "" ) -> None:
        self._fwd_code = fwd_code
        self._bwd_code = bwd_code
        self._includes = { "*": includes } if isinstance( includes, list ) else ( includes or {} )
        self._header   = { "*": header }   if isinstance( header,   str  ) else ( header   or {} )
        self._name     = name

    def header_for( self, pass_name: str ) -> str:
        return self._header.get( "*", "" ) + self._header.get( pass_name, "" )

    def includes_for( self, pass_name: str ) -> list[ str ]:
        return self._includes.get( "*", [] ) + self._includes.get( pass_name, [] )

    def signature( self ) -> str:
        parts = [ self._fwd_code, self._bwd_code, repr( self._header ), repr( self._includes ) ]
        if self._name:
            parts.insert( 0, self._name )
        return "__".join( parts )

    def code_for( self, pass_name: str ) -> str:
        if pass_name == "fwd":
            return self._fwd_code
        if pass_name == "bwd":
            return self._bwd_code
        return ""

    @property
    def has_grad_code( self ) -> bool:
        return bool( self._bwd_code )

    @property
    def name( self ) -> str:
        return self._name


class FfiCodeParallel( FfiCode ):
    """
    Trivially parallel FfiCode: iterates over cartesian_product_ranges of batch_axes,
    one independent call per element — no synchronisation, auto-dispatched to GPU.

    Supports vmap via with_prepended_batch_axis().
    """

    def __init__( self, batch_axes: list[ str ], fwd_body: str, bwd_body: str = "",
                  includes: "list | dict[ str, list ]" = None,
                  name: str = "" ) -> None:
        self._batch_axes = list( batch_axes )
        self._fwd_body   = fwd_body
        self._bwd_body   = bwd_body
        self._includes   = { "*": includes } if isinstance( includes, list ) else ( includes or {} )
        self._name       = name

    def with_prepended_batch_axis( self, name: str ) -> 'FfiCodeParallel':
        return FfiCodeParallel(
            batch_axes = [ name ] + self._batch_axes,
            fwd_body   = self._fwd_body,
            bwd_body   = self._bwd_body,
            name       = self._name,
        )

    def header_for( self, pass_name: str ) -> str:
        if pass_name == "fwd":
            return f"""
                struct ParallelFwd {{
                    template<class BI,class P> HD void operator()( BI batch_index, P &&p ) const {{
                        { self._fwd_body }
                    }}
                }};
            """

        if pass_name == "bwd" and self._bwd_body:
            return f"""
                struct ParallelBwd {{
                    template<class BI,class P> HD void operator()( BI batch_index, P &&p ) const {{
                        { self._bwd_body }
                    }}
                }};
            """

        return ""

    def includes_for( self, pass_name: str ) -> list[ str ]:
        return self._includes.get( "*", [] ) + self._includes.get( pass_name, [] )

    def signature( self ) -> str:
        parts = [ repr( self._batch_axes ), self._fwd_body, self._bwd_body ]
        if self._name:
            parts.insert( 0, self._name )
        return "__".join( parts )

    def code_for( self, pass_name: str ) -> str:
        batch_sizes = ", ".join( self._batch_axes )
        if pass_name == "fwd":
            return f"run_parallel( cartesian_product_ranges( tuple( { batch_sizes } ) ), ParallelFwd(), p );"
        if pass_name == "bwd" and self._bwd_body:
            return f"run_parallel( cartesian_product_ranges( tuple( { batch_sizes } ) ), ParallelBwd(), p );"
        return ""

    @property
    def has_grad_code( self ) -> bool:
        return bool( self._bwd_body )

    @property
    def name( self ) -> str:
        return self._name
