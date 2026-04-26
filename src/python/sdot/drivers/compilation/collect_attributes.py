import types as _types

class Annotation:
    def __init__( self, value ):
        self.value = value

def collect_attributes_inst( obj, add_dict = False, use_annotations = False ) -> list[ tuple[ str, any ] ]:
    res = []
    for name, value in collect_attributes( type( obj ), obj.__dict__, use_annotations = use_annotations ):
        if isinstance( value, Annotation ):
            res.append( ( name, value.value ) )
        else:
            res.append( ( name, value ) )
    return res

_NON_VALUES = ( classmethod, staticmethod, property, type, _types.FunctionType, _types.ModuleType )

def collect_attributes( cls, dct = None, use_annotations = False ) -> list[ tuple[ str, any ] ]:
    name_indices = {} # if attribute appears in a subclass and in a parent class, we want to take
    res = []

    def add( name, value ):
        if isinstance( value, _NON_VALUES ):
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

    if use_annotations:
        for name, annotation in getattr( cls, '__annotations__', {} ).items():
            add( name, Annotation( annotation ) )

    if dct is not None:
        for name, value in dct.items():
            add( name, value )

    return res

