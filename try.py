import torch
# import sdot.driver
import sdot

input_data = torch.ones((4,8))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_data = input_data.to(device)


print(torch.__version__)  # Doit afficher 2.10.0
print(torch.cuda.is_available())  # Doit retourner True si CUDA est disponible
print(torch.version.cuda)  # Doit afficher la version de CUDA compatible

# import jax
# print(jax.__version__)  # Doit afficher 0.4.23
# print(jax.devices())  # Doit lister les devices disponibles (GPU/TPU)


# Force PyTorch à utiliser le CPU
input_data = torch.ones((4, 8), device=device)


# Force le driver à utiliser le CPU
sdot.driver.user_device = device
sdot.driver.framework = "pytorch"
sdot.driver.user_dtype = "float32"

# Crée des données d'entrée minimales
dirac_xs = torch.randn(1, 10, 2, device=device)
dirac_ws = torch.randn(1, 10, device=device)
point_xs = torch.randn(1, 100, device=device)
point_ys = torch.randn(1, 100, device=device)

# Appelle la fonction problématique
try:
    from sdot.drivers.pytorch_functions.ot_plan_for_Piecewise1dAffineFunctions import SDOTFunction
    distances, barycenters, potentials, cuts = SDOTFunction.apply(dirac_xs, dirac_ws, point_xs, point_ys)
    print("Test réussi !")
except Exception as e:
    print(f"Erreur : {e}")