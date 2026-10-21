import numpy as np
import sdot

rng = np.random.default_rng( 0 )
positions = rng.random( ( 50, 2 ) )
pd = sdot.PowerDiagram( positions, boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )

cx = pd.cell_complex()
print( cx )                                    # CellComplex( dim = 2, ... vertices, ... edges, 50 cells )

# one face at a time: each facet once, with the seeds on both sides
for f in cx.facets[ :3 ]:
    a, b = f.sides                             # b < 0: the wall -1 - b of the domain ( cx.walls )
    print( f, "between", a, "and", b if b >= 0 else f"wall { -1 - b }", "length", f.measure )

# walking: down to the children, up to the parents, or straight to any dimension
c = cx.cells[ 0 ]                              # the cell of seed 0
edge = c.children[ 0 ]                         # one of its edges ( its facets, in 2D )
print( edge.vertices, edge.parents )           # its two ends, the cells on both sides
print( [ v.cells for v in c.vertices ] )       # the cells around each vertex of cell 0
neighbors = { s for f in c.children for s in f.sides if s >= 0 and s != 0 }

# all at once, as arrays: the Laguerre graph and the finite-volume transmissibilities |facet| / |p_j - p_i|
sides = cx.facets.sides
inner = sides[ :, 1 ] >= 0
i, j = sides[ inner ].T
trans = cx.facets.measures[ inner ] / np.linalg.norm( cx.facets.seed_vectors[ inner ], axis = 1 )

# incidences as CSR arrays: the vertices of cell k are ids[ off[ k ] : off[ k + 1 ] ], in counterclockwise order
off, ids = cx.cells.incident( 0 )
polygon = cx.vertex_positions[ ids[ off[ 0 ] : off[ 1 ] ] ]
assert abs( cx.cells.measures.sum() - 1 ) < 1e-5
