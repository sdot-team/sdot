# """
# BSP construit de façon streaming avec Dask — aucun chargement global en RAM.

# Phases :
#   1. Passe rapide sur les chunks : calcul du bbox global (min/max par axe)
#   2. Construction de l'arbre BSP depuis bbox + N total (pas de données en RAM)
#      Splits au milieu de la plus longue dimension, jusqu'à ce que le volume
#      estimé par feuille corresponde à max_leaf_bytes.
#   3. Passe d'affectation (lazy) : chaque chunk → dict {leaf_id → points_chunk}
#   4. Par feuille : concat lazy des contributions de tous les chunks → traitement

# À terme : le BSP sera en C++, cette version sert à valider l'architecture.
# """

# from __future__ import annotations

# import dask
# import dask.array as da  # type: ignore[import-untyped]
# import numpy as np

# # ---------------------------------------------------------------------------
# # Paramètres
# # ------------------------------------------ ---------------------------------
# DIM = 3
# N_POINTS = 1_000_000
# DTYPE = np.float32
# BYTES_PER_POINT = DIM * np.dtype( DTYPE ).itemsize  # 12 octets pour float32 3D
# MAX_LEAF_BYTES = 4 * 1024  # 4 Ko par feuille
# MAX_LEAF_POINTS = MAX_LEAF_BYTES // BYTES_PER_POINT  # ~341 points/feuille
# CHUNK_SIZE = 100_000


# # ---------------------------------------------------------------------------
# # 1. Génération / lecture des points
# # ---------------------------------------------------------------------------


# def make_random_points_dask( n: int, dim: int, chunk: int ) -> da.Array:
#     """Génère n points float32 aléatoires dans [0,1)^dim comme tableau Dask."""
#     return da.random.random( ( n, dim ), chunks = ( chunk, dim ) ).astype( DTYPE )


# def read_binary_points_dask(filepath: str, dim: int, chunk: int) -> da.Array:
#     """
#     Lecture lazy d'un fichier binaire float32 via Dask.
#     Le fichier contient des points consécutifs : x0 y0 z0 x1 y1 z1 ...
#     """

#     @dask.delayed
#     def _load(path: str) -> np.ndarray:
#         raw = np.fromfile(path, dtype=np.float32)
#         return raw.reshape(-1, dim)

#     # Pour da.from_delayed on a besoin de la shape exacte.
#     # Si elle n'est pas connue à l'avance, lire juste les 8 premiers octets
#     # d'un header, ou accepter un scan initial.
#     @dask.delayed
#     def _shape(path: str) -> tuple[int, int]:
#         n = np.fromfile(path, dtype=np.float32).size // dim
#         return (n, dim)

#     loaded = _load(filepath)
#     shape = dask.compute(_shape(filepath))[0]
#     return da.from_delayed(loaded, shape=shape, dtype=np.float32)


# # ---------------------------------------------------------------------------
# # 2. Arbre BSP — structure uniquement, sans données
# # ---------------------------------------------------------------------------


# class BspNode:
#     """Nœud d'un BSP axis-aligned. Les feuilles stockent un leaf_id."""

#     __slots__ = ("bbox_min", "bbox_max", "axis", "split_val", "left", "right", "leaf_id")

#     def __init__(self, bbox_min: np.ndarray, bbox_max: np.ndarray):
#         self.bbox_min: np.ndarray = bbox_min
#         self.bbox_max: np.ndarray = bbox_max
#         self.axis: int = -1
#         self.split_val: float = 0.0
#         self.left: BspNode | None = None
#         self.right: BspNode | None = None
#         self.leaf_id: int = -1


# def build_bsp_structure(
#     bbox_min: np.ndarray,
#     bbox_max: np.ndarray,
#     total_n: int,
#     max_leaf_points: int,
# ) -> BspNode:
#     """
#     Construit la structure de l'arbre sans données.
#     Split au milieu de l'axe le plus long, estimation uniforme des counts.
#     """
#     root = BspNode(bbox_min.copy(), bbox_max.copy())

#     def _split(node: BspNode, est_n: int, counter: list[int]) -> None:
#         if est_n <= max_leaf_points:
#             node.leaf_id = counter[0]
#             counter[0] += 1
#             return
#         extents = node.bbox_max - node.bbox_min
#         axis = int(np.argmax(extents))
#         mid = (node.bbox_min[axis] + node.bbox_max[axis]) / 2.0

#         node.axis = axis
#         node.split_val = float(mid)

#         bmax_l = node.bbox_max.copy()
#         bmax_l[axis] = mid
#         bmin_r = node.bbox_min.copy()
#         bmin_r[axis] = mid

#         node.left = BspNode(node.bbox_min.copy(), bmax_l)
#         node.right = BspNode(bmin_r, node.bbox_max.copy())

#         half = est_n // 2
#         _split(node.left, half, counter)
#         _split(node.right, est_n - half, counter)

#     _split(root, total_n, [0])
#     return root


# def count_leaves(root: BspNode) -> int:
#     if root.leaf_id >= 0:
#         return 1
#     assert root.left and root.right
#     return count_leaves(root.left) + count_leaves(root.right)


# # ---------------------------------------------------------------------------
# # 3. Affectation streaming : chunk → {leaf_id: points}
# # ---------------------------------------------------------------------------


# def assign_chunk(chunk: np.ndarray, root: BspNode) -> dict[int, np.ndarray]:
#     """
#     Affecte les points d'un chunk numpy à leurs feuilles BSP.
#     Retourne {leaf_id: sous-tableau de points}.
#     Fonctionne en O(n * profondeur) sans copie globale.
#     """
#     leaf_ids = np.empty(len(chunk), dtype=np.int32)

#     def _walk(node: BspNode, idx: np.ndarray) -> None:
#         if len(idx) == 0:
#             return
#         if node.leaf_id >= 0:
#             leaf_ids[idx] = node.leaf_id
#             return
#         assert node.left and node.right
#         mask_l = chunk[idx, node.axis] <= node.split_val
#         _walk(node.left, idx[mask_l])
#         _walk(node.right, idx[~mask_l])

#     _walk(root, np.arange(len(chunk), dtype=np.int32))

#     result: dict[int, np.ndarray] = {}
#     for lid in np.unique(leaf_ids):
#         result[int(lid)] = chunk[leaf_ids == lid]
#     return result


# # ---------------------------------------------------------------------------
# # 4. Pipeline Dask complet
# # ---------------------------------------------------------------------------


# def bsp_pipeline(pts_dask: da.Array, max_leaf_points: int):
#     """
#     Pipeline entièrement lazy :
#       - Phase 1 : bbox global (passe légère sur les chunks)
#       - Phase 2 : construction de l'arbre (pur Python, instantané)
#       - Phase 3 : graph Dask d'affectation des chunks
#       - Phase 4 : graph Dask de concat par feuille

#     Retourne une liste de `dask.delayed` numpy arrays, un par feuille.
#     Rien n'est calculé avant le premier .compute().
#     """
#     dim = pts_dask.shape[1]

#     # -- Phase 1 : bbox (deux réductions légères, ~O(N) lectures séquentielles)
#     print("Phase 1 : calcul du bbox global …")
#     bbox_min, bbox_max = dask.compute(
#         pts_dask.min(axis=0),
#         pts_dask.max(axis=0),
#     )
#     print(f"  bbox_min={bbox_min}, bbox_max={bbox_max}")

#     # -- Phase 2 : arbre (pas de données)
#     total_n = pts_dask.shape[0]
#     root = build_bsp_structure(bbox_min, bbox_max, total_n, max_leaf_points)
#     n_leaves = count_leaves(root)
#     print(f"Phase 2 : arbre BSP → {n_leaves} feuilles estimées")

#     # -- Phase 3 : affectation lazy chunk par chunk
#     # pts_dask.to_delayed() → ndarray (n_row_chunks, 1) de delayed
#     delayed_chunks = [
#         col[0]  # squeeze dim colonnes
#         for col in pts_dask.to_delayed()
#     ]

#     assign = dask.delayed(assign_chunk, pure=True)
#     chunk_dicts = [assign(c, root) for c in delayed_chunks]

#     # -- Phase 4 : par feuille, concat des morceaux issus de tous les chunks
#     @dask.delayed
#     def _get_part(d: dict[int, np.ndarray], lid: int, d_col: int) -> np.ndarray:
#         return d.get(lid, np.empty((0, d_col), dtype=DTYPE))

#     @dask.delayed
#     def _concat(parts: list[np.ndarray]) -> np.ndarray:
#         return np.concatenate(parts, axis=0)

#     leaf_arrays = []
#     for lid in range(n_leaves):
#         parts = [_get_part(cd, lid, dim) for cd in chunk_dicts]
#         leaf_arrays.append(_concat(parts))

#     return root, leaf_arrays


# # ---------------------------------------------------------------------------
# # 5. Traitement des feuilles (placeholder → remplacer par sdot)
# # ---------------------------------------------------------------------------


# @dask.delayed
# def process_leaf(pts: np.ndarray) -> dict:
#     """Placeholder : stats par feuille. Remplacer par un appel sdot."""
#     if len(pts) == 0:
#         return {"n": 0, "centroid": None}
#     return {"n": len(pts), "centroid": pts.mean(axis=0)}


# # ---------------------------------------------------------------------------
# # 6. Main
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     pts_dask = make_random_points_dask(N_POINTS, DIM, CHUNK_SIZE)
#     print(f"Tableau Dask : {pts_dask}")
#     print(f"MAX_LEAF_BYTES={MAX_LEAF_BYTES} B → max {MAX_LEAF_POINTS} pts/feuille\n")

#     root, leaf_arrays = bsp_pipeline(pts_dask, MAX_LEAF_POINTS)

#     print(f"\nPhase 5 : traitement de {len(leaf_arrays)} feuilles …")
#     results = dask.compute(*[process_leaf(la) for la in leaf_arrays])

#     sizes = [r["n"] for r in results]
#     print(f"Feuilles non vides : {sum(1 for s in sizes if s > 0)}")
#     print(f"Points/feuille : min={min(sizes)}, max={max(sizes)}, moy={np.mean(sizes):.1f}")
#     print(f"Total points traités : {sum(sizes):,}")
#     print("\nDone.")

from sdot.bindings import sdot_bsp_bindings
import dask.array as da  # type: ignore[import-untyped]
import numpy
import dask

def tree_reduce( list, func ):
    while len( list ) > 1:
        next_level = []
        for i in range( 0, len( list ), 2 ):
            if i + 1 < len( list ):
                next_level.append( func( list[ i ], list[ i + 1 ] ) )
            else:
                next_level.append( list[ i ] )
        list = next_level
    return list[ 0 ]

def make_bsp_set( delayed_chunks, min_bsp_count = 2, max_bsp_size = 1e6 ):
    Bsp = sdot_bsp_bindings.Bsp_FP64
    dim = 2

    # make nb_bsps void trees
    bsps = [ dask.delayed( Bsp )( dim ) ]

    # summary of correlation matrix updates
    while True:
        corr = [ dask.delayed( Bsp.avg_data_for )( bsp, chunk ) for bsp in bsps for chunk in delayed_chunks ]
        rcor = tree_reduce( corr, dask.delayed( Bsp.avg_reduction ) )
        print( rcor.compute() )
        break


    # return rcor.compute()


    # # @dask.delayed
    # # def make_bsp_and_update( chunk ):
    # #     bsp = ()
    # #     change = bsp.update_corr_matrix( chunk )
    # #     return bsp, change

    # # en première étape, on fait des BSP sans points
    # # une fois que les points logent en RAM, on peut appeler add_point et finalize( chunks ) sur chaque BSP
    # #  -> est-ce qu'on pourrait directement faire le bon nombre de BSP, par exemple en fonction du hardware et de la taille ?

    # # 1. on fait un Bsp vide pour chaque chunk
    # @dask.delayed
    # def make_bsp_and_update( chunk ):
    #     bsp = sdot_bsp_bindings.Bsp_FP64()
    #     change = bsp.update_corr_matrix( chunk )
    #     return bsp, change

    # bsp_change_pairs = [ make_bsp_and_update( c ) for c in delayed_chunks ]

    # # 2. Réduction en arbre des "change" ; l'instance BSP du premier opérande est conservée

    # root_pair = tree_reduce( bsp_change_pairs )

    # @dask.delayed
    # def get_final_change( pair ):
    #     _, change = pair
    #     return change

    # final_change = get_final_change( root_pair )

    # # 3. Appliquer le change final sur toutes les instances BSP
    # @dask.delayed
    # def apply_to_bsp( pair, change ):
    #     bsp, _ = pair
    #     bsp.apply_change( change )
    #     return bsp

    # final_bsps = [ apply_to_bsp( pair, final_change ) for pair in bsp_change_pairs ]

    # # Déclencher le calcul
    # bsps = dask.compute( *final_bsps )
    # print( bsps[ 0 ] )

points = da.random.random( ( 30, 2 ), chunks = ( 10, 2 ) ).astype( "float64" )
bspset = make_bsp_set( [ blk[ 0 ] for blk in points.to_delayed() ] )

print( bspset )
