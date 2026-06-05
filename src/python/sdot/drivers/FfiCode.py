from textwrap import dedent, indent

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
                  per_thread_args: "list[ str ] | dict[ str, list[ str ] ]" = None,
                  per_thread: "str | dict[ str, str ]" = None,
                  includes: "list | dict[ str, list ]" = None,
                  name: str = "" ) -> None:
        self._per_thread_args = { "*": per_thread_args } if isinstance( per_thread_args, list ) else ( per_thread_args or {} )
        self._per_thread      = { "*": per_thread } if isinstance( per_thread, str ) else ( per_thread or {} )
        self._batch_axes      = list( batch_axes )
        self._includes        = { "*": includes } if isinstance( includes, list ) else ( includes or {} )
        self._fwd_body        = fwd_body
        self._bwd_body        = bwd_body
        self._name            = name

    def with_prepended_batch_axis( self, name: str ) -> 'FfiCodeParallel':
        return FfiCodeParallel(
            per_thread_args = self._per_thread_args,
            per_thread      = self._per_thread,
            batch_axes      = [ name ] + self._batch_axes,
            includes        = self._includes,
            fwd_body        = self._fwd_body,
            bwd_body        = self._bwd_body,
            name            = self._name,
        )

    def header_for( self, pass_name: str ) -> str:
        lines = []
        lines.append( f"struct Parallel_{ pass_name } {{" )

        # per_thread
        if pt := self.per_thread_for( pass_name ):
            lines.append( "    template<class A,class B,class C,class D>" )
            lines.append( "    HD void per_thread( const A &thread_info, const B &batch_indices, C &&cont, D &&p ) const {" )
            lines.append( indent( dedent( pt ), '        ' )  )
            lines.append( "    }" )

        pa = self.per_thread_args_for( pass_name )

        # operator()
        args = [ "BI batch_index", "P &&p" ] + [ f"T_{ a } { a }" for a in pa ]
        prms = [ "class BI", "class P" ] + [ f"class T_{ a }" for a in pa ]
        lines.append( f"    template<{ ', '.join( prms ) }> HD void operator()( { ', '.join( args ) } ) const {{" )
        if pass_name == "fwd":
            lines.append( self._fwd_body )
        if pass_name == "bwd" and self._bwd_body:
            lines.append( self._bwd_body )
        lines.append( "    }" )

        lines.append( "};" )
        return "\n".join( lines )

    def includes_for( self, pass_name: str ) -> list[ str ]:
        return self._includes.get( "*", [] ) + self._includes.get( pass_name, [] )

    def per_thread_args_for( self, pass_name: str ) -> list[ str ]:
        if pass_name in self._per_thread_args:
            return self._per_thread_args[ pass_name ]
        if "*" in self._per_thread_args:
            return self._per_thread_args[ "*" ]
        return []

    def per_thread_for( self, pass_name: str ) -> str:
        if pass_name in self._per_thread:
            return self._per_thread[ pass_name ]
        if "*" in self._per_thread:
            return self._per_thread[ "*" ]
        return ""

    def signature( self ) -> str:
        parts = [ self._fwd_body, self._bwd_body, repr( self._batch_axes ), repr( self._includes ), repr( self._per_thread ),  ]
        if self._name:
            parts.insert( 0, self._name )
        return "__".join( parts )

    def code_for( self, pass_name: str ) -> str:
        batch_sizes = ", ".join( self._batch_axes )
        return f"run_parallel( cartesian_product_ranges( tuple( { batch_sizes } ) ), Parallel_{ pass_name }(), p );"

    @property
    def has_grad_code( self ) -> bool:
        return bool( self._bwd_body )

    @property
    def name( self ) -> str:
        return self._name
