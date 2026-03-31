import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print('CUDA available ', torch.cuda.is_available() )
print('CUDA Version: ', torch.version.cuda)

def f(x):
    return  x**2

# Créer un tenseur x avec requires_grad=True pour permettre le calcul des gradients
x = torch.tensor(3.0, requires_grad=True, device=device)
y = f(x)
y.backward()

# Afficher la dérivée
print(f"La dérivée de f(x) = x² en x = {x.item()} est {x.grad.item()}")
print("point derivatives of ", y.grad_fn)
print("-"*60)
print("Example with f(x) = (x² + 1)² ")

x = torch.tensor(3.0, requires_grad=True)
a = x**2
b = a + 1
y = b**2

# print("y = b**2 grad ", y.grad_fn)
# print(y.grad_fn.next_functions)
# print(y.grad_fn.next_functions[0][0])
# print(y.grad_fn.next_functions[0][0].next_functions)
# print(y.grad_fn.next_functions[0][0].next_functions[0][0])
# print(y.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
# y.backward()
# print(x.grad)

""" AUTOGRAD pytorch implicite """

class SquareFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)# on sauvegarde x pour backward
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        # récupérer x sauvegardé
        x, = ctx.saved_tensors
        grad_x = grad_output * 2 * x    # dérivée locale = 2x
        return grad_x

target = torch.tensor(5.0)
x = torch.tensor(3.0, requires_grad=True, device=device)
y = SquareFunction.apply(x)
L = (y - target) ** 2
dL_dy = 2 * (y - target)
x.register_hook(lambda g: print(g)) # Tu vois passer :le gradient réel.
with torch.autograd.profiler.profile() as prof:
    y.backward(dL_dy, retain_graph=True) # sinon  graphe est détruit après le premier backward.
print(prof.key_averages())
print("x device:", x.device)
print("x =", x.item())
print("y =", y.item())
print("L = (y - target) ** 2 : ",((y - target).item()) ** 2)
print("L =", L.item())
print("dL_dy = 2 * (y - target)",  2 * (y - target).item())
print("x.grad ou dL/dx =",  x.grad.item())

grad_x = torch.autograd.grad(y, x, grad_outputs=dL_dy, retain_graph=True)
print("torch.autograd.grad", grad_x[0].item())
# x.grad = None
x.grad.zero_()
"""
Pourquoi PyTorch préfère None
Parce qu’au backward suivant :
PyTorch peut optimiser l’allocation.
x.grad.zero_()
garde le tensor : tensor existe encore


optimizer.zero_grad() Cela remet tous les gradients à zéro.
ntuition profonde
backward accumule :
48+48=96
48+48=96
grad retourne :
48
48

sans accumulation.

1️⃣2️⃣ Donc règle scientifique

Pour vérifier maths :

✅ torch.autograd.grad

Pour entraînement :

✅ backward + zero_grad
"""
y.backward(dL_dy)
print("x.grad after 2d backward", x.grad.item())

#
"""
7️⃣ Règle mentale ultra importante si pas retain_graph=True
backward :
consomme graphe
+
accumule gradient

grad :
consomme graphe
+
retourne gradient
"""



#
# # Définir une couche linéaire simple
# class SimpleLinearLayer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SimpleLinearLayer, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.linear(x)
#
# ""
# print("couche linéaire # Cible (pour f(x) = 3x, y_true devrait être 6 si x=2)")
# # Instanciation de la couche
# input_dim = 1
# output_dim = 1
# learning_rate = 0.01
# layer = SimpleLinearLayer(input_dim, output_dim)
#
# # Définir une entrée et une cible
# x = torch.tensor([[2.0]], requires_grad=False)  # Entrée
# y_true = torch.tensor([[6.0]])  # Cible (pour f(x) = 3x, y_true devrait être 6 si x=2)
#
# # Forward pass
# y_pred = layer(x)
#
# # Définir une fonction de perte (MSE)
# loss_fn = nn.MSELoss()
# loss = loss_fn(y_pred, y_true)
#
# # Nettoyer les gradients existants
# layer.zero_grad()
#
# # Backward pass pour calculer les gradients
# loss.backward()
#
# # Afficher les poids et leurs gradients avant la mise à jour
# print("Avant la mise à jour:")
# print(f"Poids: {layer.linear.weight.data}")
# print(f"Biais: {layer.linear.bias.data}")
# print(f"Gradient des poids: {layer.linear.weight.grad}")
# print(f"Gradient du biais: {layer.linear.bias.grad}")
#
# # Mise à jour manuelle des poids et biais
# with torch.no_grad():
#     layer.linear.weight -= learning_rate * layer.linear.weight.grad
#     layer.linear.bias -= learning_rate * layer.linear.bias.grad
#
# # Afficher les poids et leurs gradients après la mise à jour
# print("\nAprès la mise à jour:")
# print(f"Poids: {layer.linear.weight.data}")
# print(f"Biais: {layer.linear.bias.data}")
#
# # Refaire un forward pass pour voir l'effet de la mise à jour
# y_pred_updated = layer(x)
# print(f"\nPrédiction après mise à jour: {y_pred_updated.item()}")
#
# """ BACKWARD explicite """
#
# # Définir une fonction personnalisée avec backward explicite
# class LinearFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight, bias):
#         ctx.save_for_backward(x, weight, bias)
#         return weight * x + bias
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, weight, bias = ctx.saved_tensors
#         grad_x = grad_output * weight
#         grad_weight = grad_output * x
#         grad_bias = grad_output
#         return grad_x, grad_weight, grad_bias
#
# # Définir une couche personnalisée utilisant LinearFunction
# class CustomLinearLayer(torch.nn.Module):
#     def __init__(self):
#         super(CustomLinearLayer, self).__init__()
#         self.weight = torch.nn.Parameter(torch.tensor(1.0))  # Initialiser le poids
#         self.bias = torch.nn.Parameter(torch.tensor(0.0))    # Initialiser le biais
#
#     def forward(self, x):
#         return LinearFunction.apply(x, self.weight, self.bias)
#
# # Instanciation de la couche personnalisée
# layer = CustomLinearLayer()
#
# # Définir une entrée et une cible
# x = torch.tensor(2.0, requires_grad=False)  # Entrée
# y_true = torch.tensor(4.0)  # Cible (pour f(x) = 2x, y_true devrait être 4 si x=2)
#
# # Forward pass
# y_pred = layer(torch.tensor([x]))
#
# # Définir une fonction de perte (MSE)
# loss_fn = torch.nn.MSELoss()
# loss = loss_fn(y_pred, torch.tensor([y_true]))
#
# # Nettoyer les gradients existants
# layer.zero_grad()
#
# # Backward pass pour calculer les gradients
# loss.backward()
#
# # Afficher les poids et leurs gradients avant la mise à jour
# print("Avant la mise à jour:")
# print(f"Poids: {layer.weight.data}")
# print(f"Biais: {layer.bias.data}")
# print(f"Gradient des poids: {layer.weight.grad}")
# print(f"Gradient du biais: {layer.bias.grad}")
#
# # Définir un taux d'apprentissage
# learning_rate = 0.01
#
# # Mise à jour manuelle des poids et biais
# with torch.no_grad():
#     layer.weight -= learning_rate * layer.weight.grad
#     layer.bias -= learning_rate * layer.bias.grad
#
# # Afficher les poids et leurs gradients après la mise à jour
# print("\nAprès la mise à jour:")
# print(f"Poids: {layer.weight.data}")
# print(f"Biais: {layer.bias.data}")
#
# # Refaire un forward pass pour voir l'effet de la mise à jour
# y_pred_updated = layer(torch.tensor([x]))
# print(f"\nPrédiction après mise à jour: {y_pred_updated.item()}")