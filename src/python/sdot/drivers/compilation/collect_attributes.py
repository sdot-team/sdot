class Annotation:
    def __init__( self, value ):
        self.value = value

def collect_attributes_inst( obj, add_dict = False ) -> list[ tuple[ str, any ] ]:
    return collect_attributes( type( obj ), obj.__dict__ )

def collect_attributes( cls, dct = None ) -> list[ tuple[ str, any ] ]:
    res = []
    name_indices = {} # if attribute appears in a subclass and in a parent class, we want to take

    def add( name, value ):
        if isinstance( value, ( classmethod, staticmethod ) ) or callable( value ):
            return
        if isinstance( value, property ): #  and value.fset is None
            return
        if name.startswith( '_' ):
            return

        # register value
        if name in name_indices:
            res[ name_indices[ name ] ] = ( name, value )
        else:
            name_indices[ name ] = len( res )
            res.append( ( name, value ) )

    for klass in reversed( cls.__mro__ ):
        for name, value in vars( klass ).items():
            add( name, value )
        for name, annotation in getattr( cls, '__annotations__', {} ).items():
            add( name, Annotation( annotation ) )

    if dct is not None:
        for name, value in dct.items():
            add( name, value )

    return res

