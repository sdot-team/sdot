import torch
import torch.nn as nn

# Définir la fonction f(x) = 2x
def f(x):
    return  x**2

# Créer un tenseur x avec requires_grad=True pour permettre le calcul des gradients
x = torch.tensor(3.0, requires_grad=True)

# Calculer f(x)
y = f(x)

# Calculer la dérivée de y par rapport à x
y.backward()

# Afficher la dérivée
print(f"La dérivée de f(x) = 2x en x = {x.item()} est {x.grad.item()}")

""" AUTOGRAD pytorch implicite """


# Définir une couche linéaire simple
class SimpleLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

""
print("couche linéaire # Cible (pour f(x) = 3x, y_true devrait être 6 si x=2)")
# Instanciation de la couche
input_dim = 1
output_dim = 1
learning_rate = 0.01
layer = SimpleLinearLayer(input_dim, output_dim)

# Définir une entrée et une cible
x = torch.tensor([[2.0]], requires_grad=False)  # Entrée
y_true = torch.tensor([[6.0]])  # Cible (pour f(x) = 3x, y_true devrait être 6 si x=2)

# Forward pass
y_pred = layer(x)

# Définir une fonction de perte (MSE)
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)

# Nettoyer les gradients existants
layer.zero_grad()

# Backward pass pour calculer les gradients
loss.backward()

# Afficher les poids et leurs gradients avant la mise à jour
print("Avant la mise à jour:")
print(f"Poids: {layer.linear.weight.data}")
print(f"Biais: {layer.linear.bias.data}")
print(f"Gradient des poids: {layer.linear.weight.grad}")
print(f"Gradient du biais: {layer.linear.bias.grad}")

# Mise à jour manuelle des poids et biais
with torch.no_grad():
    layer.linear.weight -= learning_rate * layer.linear.weight.grad
    layer.linear.bias -= learning_rate * layer.linear.bias.grad

# Afficher les poids et leurs gradients après la mise à jour
print("\nAprès la mise à jour:")
print(f"Poids: {layer.linear.weight.data}")
print(f"Biais: {layer.linear.bias.data}")

# Refaire un forward pass pour voir l'effet de la mise à jour
y_pred_updated = layer(x)
print(f"\nPrédiction après mise à jour: {y_pred_updated.item()}")

""" BACKWARD explicite """

# Définir une fonction personnalisée avec backward explicite
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        return weight * x + bias

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_output * weight
        grad_weight = grad_output * x
        grad_bias = grad_output
        return grad_x, grad_weight, grad_bias

# Définir une couche personnalisée utilisant LinearFunction
class CustomLinearLayer(torch.nn.Module):
    def __init__(self):
        super(CustomLinearLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))  # Initialiser le poids
        self.bias = torch.nn.Parameter(torch.tensor(0.0))    # Initialiser le biais

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)

# Instanciation de la couche personnalisée
layer = CustomLinearLayer()

# Définir une entrée et une cible
x = torch.tensor(2.0, requires_grad=False)  # Entrée
y_true = torch.tensor(4.0)  # Cible (pour f(x) = 2x, y_true devrait être 4 si x=2)

# Forward pass
y_pred = layer(torch.tensor([x]))

# Définir une fonction de perte (MSE)
loss_fn = torch.nn.MSELoss()
loss = loss_fn(y_pred, torch.tensor([y_true]))

# Nettoyer les gradients existants
layer.zero_grad()

# Backward pass pour calculer les gradients
loss.backward()

# Afficher les poids et leurs gradients avant la mise à jour
print("Avant la mise à jour:")
print(f"Poids: {layer.weight.data}")
print(f"Biais: {layer.bias.data}")
print(f"Gradient des poids: {layer.weight.grad}")
print(f"Gradient du biais: {layer.bias.grad}")

# Définir un taux d'apprentissage
learning_rate = 0.01

# Mise à jour manuelle des poids et biais
with torch.no_grad():
    layer.weight -= learning_rate * layer.weight.grad
    layer.bias -= learning_rate * layer.bias.grad

# Afficher les poids et leurs gradients après la mise à jour
print("\nAprès la mise à jour:")
print(f"Poids: {layer.weight.data}")
print(f"Biais: {layer.bias.data}")

# Refaire un forward pass pour voir l'effet de la mise à jour
y_pred_updated = layer(torch.tensor([x]))
print(f"\nPrédiction après mise à jour: {y_pred_updated.item()}")