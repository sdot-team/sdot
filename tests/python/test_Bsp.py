import sdot

def test_bsp():
    positions = sdot.driver.t2( [ [ 1 ], [ 2 ] ] )
    weights = sdot.driver.t1( [ 3, 4 ] )
    bsp = sdot.Bsp( positions, weights )
    info( bsp )

if __name__ == "__main__":
    test_bsp()
