

class Distribution:
    """
        Ancestor class for distributions


    """
    
    def boundary_split( self, ndim = None ):
        """ get a tuple with
              * a list of convex boundaries (can be a void list)
              * the value inside the boundaries (something that can be converted to an Expr, but without any trivial convex boundaries)
        """
        return None
    
    @property
    def ndim( self ):
        """ return the prescribed dimensionality if there's any """
        return None 
    
