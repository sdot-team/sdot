import numpy

class Plotter:
    def __init__( self, plotter = None ):
        import pyvista
        self._owned = plotter is None
        if self._owned:
            self._plotter = pyvista.Plotter( theme = pyvista.themes.DarkTheme() )
        else:
            self._plotter = plotter
        self._dim = None

    def __enter__( self ):
        return self

    def __exit__( self, exc_type, exc_val, exc_tb ):
        if self._owned and exc_type is None:
            if self._dim == 2:
                self._plotter.view_xy()
            self._plotter.reset_camera()
            self._plotter.show()

    def plot_mesh( self, points, faces ):
        import pyvista
        pts = numpy.asarray( points )

        self._dim = pts.shape[ 1 ]

        if pts.shape[ 1 ] < 3:
            pts = numpy.hstack( [ pts, numpy.zeros( ( pts.shape[ 0 ], 3 - pts.shape[ 1 ] ) ) ] )

        self._plotter.add_mesh( pyvista.PolyData( pts, faces = numpy.asarray( faces, dtype = numpy.int64 ) ), show_edges = True )
