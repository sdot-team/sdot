
def find( iterable, predicate ):
    return next( ( x for x in iterable if predicate( x ) ), None )

def index( iterable, value ):
    try:
        return iterable.index( value )
    except ValueError:
        return -1

