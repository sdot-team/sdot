from typing import cast

class CpyArg:
    def __init__( self, name: str, code: str = "", sub_list: list | None = None ):
        self.name = name
        self.code = code
        self.value = None
        self.sub_list = sub_list
        self.for_return = 0 # 0 => pure input, 1 => mutable, 2 => return
        self.signature_type = ""
        self._python_ctor: callable | tuple[ int, int, bool ] | None = None # for output value. ( 1 if differentiable, num in list, validity )

    def arg( self, name: str ):
        cpy_arg = CpyArg( name )

        cpy_arg.for_return = self.for_return

        if self.sub_list is None:
            self.sub_list = []
        self.sub_list.append( cpy_arg )

        return cpy_arg

    def reassemble( self, differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs ):
        # tensor ?
        if isinstance( self._python_ctor, tuple ):
            assert len( self._python_ctor ) == 3

            # not valid ?
            if not self._python_ctor[ 2 ]:
                return None

            # else, get tensor
            if self.for_return:
                if self._python_ctor[ 0 ]:
                    return differentiable_outputs[ self._python_ctor[ 1 ] ]
                return non_differentiable_outputs[ self._python_ctor[ 1 ] ]
            if self._python_ctor[ 0 ]:
                return differentiable_inputs[ self._python_ctor[ 1 ] ]
            return non_differentiable_inputs[ self._python_ctor[ 1 ] ]

        #
        if self._python_ctor is None:
            raise RuntimeError( f"No python ctor for { self.name }" )

        # ctor
        if self.sub_list is None:
            return self._python_ctor()
        return self._python_ctor( *[ item.reassemble( differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs ) for item in self.sub_list ] )

    def update( self, obj, cb, differentiable_output_values, non_differentiable_output_values ):
        """ obj = ref. cb = how to modify the ref """
        # tensor ?
        if isinstance( self._python_ctor, tuple ):
            assert len( self._python_ctor ) == 3

            # not valid ?
            if not self._python_ctor[ 2 ]:
                return

            if cb is None:
                raise RuntimeError( "Mutable() takes only objects that contains other objects (like lists for instance)" )
            if self._python_ctor[ 0 ]:
                return cb( differentiable_output_values[ self._python_ctor[ 1 ] ] )
            return cb( non_differentiable_output_values[ self._python_ctor[ 1 ] ] )

        # ctor
        if self.sub_list is None:
            return

        for s in self.sub_list:
            # get sub obj + how to update it
            s = cast( CpyArg, s )
            if s.name.isdigit():
                k = int( s.name )
                nobj = obj[ k ]
                def ncb( x ):
                    obj[ k ] = x
            else:
                nobj = getattr( obj, s.name )
                def ncb( x ):
                    setattr( obj, s.name, x )

            # recursive call
            s.update( nobj, ncb, differentiable_output_values, non_differentiable_output_values )


    def assembled_code( self ) -> str:
        res = self.code
        if self.sub_list is not None:
            res += "( " + str.join( ", ", [ a.assembled_code() for a in self.sub_list ] ) + " )"
        return res
