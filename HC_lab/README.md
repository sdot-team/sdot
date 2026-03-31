# ENV

make venv
- var env
export PYTHONPATH=/home/hcourtei/Projects/sdot/sdot_with_interfaces/src/python:/home/hcourtei/Projects/sdot/sdot_with_interfaces/build/src/python:$PYTHONPATH
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
- version cuda pour laptop

pip uninstall torch
pip install torch  --index-url https://download.pytorch.org/whl/cu121

`make test`
`python -m pytest -s -q --tb=short tests/ -v`
`python docs/examples/ct_reconstruction/ct_reconstruction.py`


Architecture globale de src/python/sdot
1. Modules de base
plan.py : Gère la création des plans de transport optimal entre distributions
driver.py : Module central qui gère le framework (PyTorch, JAX), device (CPU/GPU) et type de données (FP32/FP64)
2. Distributions
distributions/ : Contient les classes pour représenter différentes distributions :
SumOfWeightedDiracs : Représente des points pondérés
PiecewiseAffineFunction1d : Fonctions affines morceaux en 1D
BatchOfDistributions : Gestion des lots de distributions
3. Plans de transport
OtPlan.py : Classes de base pour les plans de transport optimal
BatchOfOtPlans.py : Gestion des lots de plans
OtPlan1d.py : Plans de transport spécifiques 1D
4. Fonctions spécifiques
drivers/pytorch_functions/ : Implémentations spécifiques pour PyTorch avec autograd
drivers/jax_functions/ : Implémentations spécifiques pour JAX
5. Bindings C++
bindings/ : Interface entre Python et code C++ optimisé pour les calculs lourds
6. Fonctions d'optimisation
optimization/ : Potentiellement pour des optimisations spécifiques
Fonctionnement
Le module principal sdot permet de :

Créer des distributions (points, fonctions)
Calculer les plans de transport optimal entre elles
Utiliser différents frameworks (PyTorch/JAX) et types de données
Effectuer des opérations avec des gradients automatiques pour validation
Le code montre une architecture modulaire qui sépare les interfaces Python des implémentations C++ optimisées pour les calculs complexes de transport optimal.


dans @ot_plan_for_Piecewise1dAffineFunctions, 

class SDOTFunction( torch.autograd.Function ): 
    def forward( ctx, dirac_xs, dirac_ws, point_xs, point_ys ) -> tuple[ torch.tensor, torch.tensor, torch.tensor, torch.tensor ]:
      ...
        getattr( sdot_bindings_cpu, "ot_plan_to_piecewise_affine_1d_" + driver.normalized_dtype )(
                dirac_xs, dirac_ws, point_xs, point_ys,
                distances, barycenters, potentials, cuts
            )


    def backward( ctx, grad_distance, grad_barycenters, grad_potentials, grad_cuts ):
