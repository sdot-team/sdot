

Prop: on ne stocke pas les edges. On va plutôt les chercher à la volée, en partant des points extérieurs.
  Rq: on pourra filtrer la liste des cuts dès le démarrage (sauf si on a besoin des recalculer les positions avec sys linéaire)
  => a priori, on n'aura plus besoin de faire de correction sur les numéros de vertices :)
  Comment trouver tous les edges qui sont attachés à un vertex ?
    Prop: on fait une table edge => numéro de noeud (qui démarre à -1)
    Lorsqu'on fait tous les edges attachés à un noeud, s'il y a déjà un noeud attaché, on a notre edge.

  Comment faire pour diminuer les mouvements du Cut et vertex ?
  
  Si on passe à une liste de vertex actifs, est-ce qu'on aurait intérêt à aggréger les données ?
    Ça pourrait être bon pour le cache...
    Il faudrait à ce compte là mettre aussi les num_cuts


Rq: on pourrait fonctionner avec par défaut des cellules "non-bornées". 
  Tant qu'elles le restent, on fait des cuts avec M\V, pour ne rater aucun point.
  Comment déterminer si les cellules sont bornées ?
    => 
  Lorsqu'elles sont bornées, on passe au cut par edge.

  
La proposition, c'est de faire un pavage pour travailler localement puis avec des faisceaux.
  Si on veut faire du périodique, on pourrait imaginer que ça soit utilisé comme paramètre du "paveur".
  Ça pourrait être la même chose pour MPI : on donne les positions pour chaque process. Le paveur se charge de déplacer les données d'un processur à l'autre
  

* On stocke la position des diracs pour chaque case
* On parcourt les cases pour construire les cellules
  * on intersecte les cellules de la case courante + l'entourage proche (à définir fonction du type de problème).
  * on va chercher s'il est possible de trouver des diracs qui peuvent intersecter la cellule, etc...

( dx / step ) * ... = target_size
( dx * dy * ... ) * step^dim = target_size
step = ( target_size / ( dx * dy * ... ) ) ^ 1/dim


e = dx * x + dy * y - do
  e vaut 0 lorsque [ x, y ] = [ dx, dy ] * do / ( dx^2 + dy^2 )

Pour la base, on propose
  b0 = c * [ dx, dy ] avec c qu'on ne connait pas

Pour la nouvelle direction, on prend
  nx = sp( b0, dir ) = c * ( dx^2 + dy^2 )

e = nx * a - no doit valoir 0 lorsque 
  b0 * a = [ dx, dy ] * do / ( dx^2 + dy^2 ), c'est à dire
  c * [ dx, dy ] * a = [ dx, dy ] * do / ( dx^2 + dy^2 ), ou
  c * a = do / ( dx^2 + dy^2 )
  a = do / ( dx^2 + dy^2 ) / c

Du coup, on doit avoir
  no = c * ( dx^2 + dy^2 ) * do / ( dx^2 + dy^2 ) / c

Rq: on aimerait bien avoir une cellule plus générique. 
  Prop 1: on travaille dans des sous-espaces
    => a priori plus rapide
    => mais il faut une base
  Prop 2: on trouve des points définis sur l'espace entier
    => ça permet 

Quand on change de dimension, les nouveaux vertices, c'est l'intersection des edges formés avec les anciens vertex et le nouveau plan de coupe
  Si les coordonnées sont exprimée dans la base, il faut juste ajouter un 0... sauf à la fin, il faut passer en coordonnées réelles


dx * x + dy * y + dz * z - off

dx * ( b_ax * a + b_bx * b ) + dy * ( b_ay * a + b_by * b ) + dz * ( b_az * a + b_bz * b ) - off

( dx * b_ax + dy * b_ay + dz * b_az )...


Rq: on pourrait systématiquement travailler en nb_dims, en imposant les 


Pb: on aimerait bien ne pas avoir à envoyer les infos de taille dynamique lorsqu'on connait la taille à priori.

Rq: pour python, on pourrait utiliser les outils habituels de production de fichiers .vtk 

L'idée de base, c'est de tester s'il est possible de trouver un dirac coupant dans les zones non explorées.
  On pourrait trouver différentes façons de délimiter l'espace.
  L'idée, c'est que s'il y a une possibilité de coupe dans cette partie de l'espace, on prend la cellule non vide la plus proche.
  Quand on teste si une zone peut contenir un dirac coupant, on est emmenés à rapetisser la zone.
  On pourra donc chercher les boites dans le résultat de cette découpe.
  A fortiori, quand on va explorer des sous-zones pour la description des poids, on va créer des subdivisions de la cellule qui décrit les zones à explorer 

Rq: on pourrait faire un "front" des cellules voisines, en mettant au début toutes les boites dont on est certains qu'elles ne peuvent plus rien couper.
  (ça utiliserait une approximation du poids par boite)
  À un moment donné, on n'aura plus que des boites non coupantes.
  Il faudra ensuite tester ce qui reste

Prop: on fait la représentation des poids de façon hiérarchique dans le grille.
  Pour chaque cellule en construction, on fait un tableau qui dit si les celules ont été parcourues complétement.
  Quand on 






