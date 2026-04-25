import numpy
import sdot

def test_bsp():
    positions = sdot.driver.t2( numpy.random.random( [ 50, 2 ] ) )
    # weights = sdot.driver.t1( [ 3, 4 ] ) , weights
    pd = sdot.PowerDiagram( positions )

    info( pd.nb_vertices_capacity )
    info( pd.nb_vertices )
    info( pd.dim )
    # info( pd.bsp.sorted_vertex_indices )
    pd.plot()

if __name__ == "__main__":
    test_bsp()
