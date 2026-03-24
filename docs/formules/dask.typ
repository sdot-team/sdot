= BSP en parallèle

Root
  -> On fait une réduction unique avec tous chunks
  -> On peut ensuite faire une autre réduction pour construire un histogram selon la meilleure direction
  -> ça permet de construire la première séparation du BSP

Ensuite, on pourrait reparcourir tous les chunks, et assigner à la volée tous les points pour calculer autant de matrice de corrélation qu'il y a de feuilles
  -> on fait du coup une réduction sur les BSP partiels
  -> il faudrait repartir à chaque étape du BSP réduit, à envoyer à chaque PC.

On est obligé de fonctionner comme ça, au moins jusqu'à ce que les points logent en mémoire
  -> on pourrait imaginer que la structure du BSP soit partagée par tout le monde, ou alors qu'on ait une partie commune pour chacun
  -> à ce moment là, on peut stocker dans


Faire un docker pour les exemples
Pas fait le code Cuda
requirement.txt
  -> et mettre les dépendances dans le pyproject.toml

Faire des paquets conda

Ajouter des nouveaux exemples


Prop pour le driver
- user_xyz correspond exactement à ce que l'user donne. Quand on le set, ça
