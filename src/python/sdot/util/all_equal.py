
def all_equal( a, b ):
    if len( a ) != len( a ):
        return False
    for va, vb in zip( a, b ):
        if va != vb:
            return False
    return True
