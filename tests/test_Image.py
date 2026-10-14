from errand import test

if test( "basic" ):
    # ds = SumOfDiracs( [ 1, 2, 3, 4 ] )
    # op = SdotPlanNd1D( ds, di )
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from loom.testing import check_grad
    di = Image( values = [[ 1, 0, 1 ]], pixel_frame = [ [ 1, 0 ], [ 0, 1 ] ] )   # pixels of 1: a mass of 2
    assert float( di.mass ) == 2
