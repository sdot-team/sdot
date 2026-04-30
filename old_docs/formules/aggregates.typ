= BSP en parallèle

La proposition avec les annotations, c'est de manipuler des "type-like" et de passer

Pour les retours, `sdot.Return( sdot.Tensor( "dim" ) )` pourrait suffire.

Objectifs suivant:
- ajouter les axes
  - une fonction


Pour chaque tenseur d'entrée, on a shape[ i ] = expr
  Si on a 3 termes, il faudrait faire toutes les combinaisons des tenseurs qui permettent d'obtenir les expressions
  En attendant, on pourrait construire les matrices input tensor -> taille
