#set heading(numbering: "1.a")
#set page(fill: black)
#set text(fill: white)

= Propositions pour les cellules non fermées

Avant que les cellules soient un espace fermé, on est amenés à prendre des décision sur des chiffres imprécis.

Si on est capable de tout le temps travailler sur des volumes fermés, on simplifie et potentiellement, on accélère.

Prop: on commence avec un volume fermée pas trop grand (pour les problèmes de précision que ça peut engendrer).

Si une coupe ne change rien, on vérifie que refaire toutes les coupes avec un volume plus grand ne change rien.


= GPU

Prop: dans le paths, on propose max sp pour tous les points
  -> il faudrait aussi garde minmax




Pb: on fait FfiArgInfo avec les tracers en en entrée
  Quand on fait la version backward, on peut conserver les tracer pour le FfiArgInfo puisque l'appel ffi se fait avec les valeurs mise à jour
