

class CellFunction:
    """
        function that take coordinate and return the value of the cell_function with the corresponding cell instance

    """

    def __init__( self, power_diagram, cell_function ):
        self.power_diagram = power_diagram
        self.cell_function = cell_function

    def __call__( self, x ):
        cell = self.power_diagram.find_cell( x )
        if cell is None:
            return None
        return self.cell_function( cell, x )

    def plot( self, plt = None ):
        if self.power_diagram.ndim != 2:
            raise RuntimeError( "TODO: plot CellFunction for dim != 2" )
        
        if plt is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig = fig.add_subplot( projection = '3d' )
            self.plot( fig )
            plt.show()
            return

        # ( self, fig, color = 'black', linestyle = '-', linewidth = 2, low_dim_linewidth_coeff = 0.25, ray_length = 0.2, ray_color = None, ray_linestyle = '--', free_color = None, free_linestyle = ':', elevation_function = None ):

        #
        def on_cell( cell ):
            cell.plot_pyplot( plt, elevation_function = lambda x: self.cell_function( cell, x ) )
        self.power_diagram.for_each_cell( on_cell, 1 )
