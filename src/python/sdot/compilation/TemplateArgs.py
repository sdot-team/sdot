from dataclasses import dataclass


@dataclass
class TemplateArg:
    cpp_type : str
    priority : int = 0


class TemplateArgs:
    """
    Ordered collection of C++ template arguments, sorted by priority at insertion time.
    Duplicate names are silently ignored (first insertion wins).
    """

    def __init__( self ):
        self._items : list[ tuple[ str, TemplateArg ] ] = []
        self._names : set[ str ]                        = set()

    def add( self, name: str, cpp_type: str, priority: int = 0 ):
        if name in self._names:
            return
        self._items.append( ( name, TemplateArg( cpp_type, priority ) ) )
        self._names.add( name )
        self._items.sort( key = lambda kv: kv[ 1 ].priority )

    def __iter__( self ):
        return iter( self._items )

    def __len__( self ):
        return len( self._items )

    def __bool__( self ):
        return bool( self._items )
