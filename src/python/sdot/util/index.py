def index( iterable, value ):
    if getattr( iterable, "index", None ) is None:
        return index( list( iterable ), value )

    try:
        return iterable.index( value )
    except ValueError:
        return -1

