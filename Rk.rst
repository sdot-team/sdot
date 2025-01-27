Urgent:
* les integrations utiles pour sdot (quantization)
  * faire des expressions pour représenter des fonctions définies par morceau
     => on pourrait faire une expression GridFunc qui prend comme argument
       * expression symbolique qui dépend de la valeur sur la cellule et sur les noeuds, par exemple
          cell_value_at[ 0, -1, 0 ]
          node_value_at[ 0, -1, 0 ]
          cell_value
          node_value

          ou alors

          cell_value_at[  ]
       * expression symbolique pour donner les valeurs quand on dépasse les bords, à base de
          "periodic", "closest", ""
       * origin
       * axes
      Autre possibilité : une formule qui va chercher dans un tableau
        tab[ floor( ( x_0 - 3 ) % 3 ) * w + clamp( floor( x_1 - 4 ), 0, h ) ] * frac( x_1 - x_0 - 4 ) + ...
        -> on repère les bornes derrière chaque floor ou frac. Dans ce cas précis, on a des fonction affines. 
        La nouvelle formule fera apparaître des valeurs connues.

Est-ce qu'on met les données à côté des expr ? C'est tentant... on pourrait proposer un Str synthétique où les valeurs n'apparaissent pas, seulement les références.



Plus tard:
* remettre l'architecture, avec option "full_opt"



Une façon de gérer des images serait de permettre de les intégrer dans les expressions symboliques
  => du coup, on pourrait faire une sortie "indépendante" (celle qui va générer le code) et une sortie "particulière", qui serait un map de valeurs en python

Prop de base pour l'intégration : on fournit une fonction pour intégrer sur les simplex...
  On pourra par exemple partir des coordonnées 


* Prop alternative: on accepte d'étaler tant que la somme est égale à la densité.
  On pourrait trouver une formule fermée en 1D non partiel

Est-ce qu'on fait un argument "entrées naturelles" ? C'est tentant...
