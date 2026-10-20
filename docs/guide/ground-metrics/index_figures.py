"""The page of `index.md`: an interactive page of the HTML viewer ( `sdot.Visualizer`, `docs/public/guide/ground-metrics-axes.html` ).

    python docs/guide/ground-metrics/index_figures.py
"""
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 2 ] / "public" / "guide"


def axes( n = 30 ):
    """two choices at once: a partial transport ( half of the square ) on a torus. The balls that cross a side come back on
    the other one, in the copies by the first ring"""
    pos = np.random.default_rng( 5 ).uniform( 0, 1, size = ( n, 2 ) )
    p = sdot.plan( sdot.SumOfDiracs( pos, target_mass = 0.5 ), sdot.Box(), periodicity = True )
    viz = sdot.Visualizer( title = "A partial transport on a torus", background = "#ffffff" )
    p.add_to_viz( viz )
    viz.write_html( OUT / "ground-metrics-axes.html" )


if __name__ == "__main__":
    OUT.mkdir( parents = True, exist_ok = True )
    axes()
