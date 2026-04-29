def index( iterable, value ):
    try:
        return iterable.index( value )
    except ValueError:
        return -1

