# Ground metrics : round 3

> Leave **Answer** empty to accept the recommendation. `?` asks for more detail.
> Write anything else (answer, objection, comment) under any question, save, then say "ok" in the chat.

Décisions mises à jour : [decisions.md](decisions.md).

---

## Q1 : (reprise, avec un exemple) Une cellule ne peut pas « dépasser » D

C'est la périodicité qui garantit que rien ne déborde : ce qui sortirait de D *est* déjà dans D, par une autre image.

Exemple 1D : D = [0, 1), un seed y = 0.9 avec un poids tel que sa cellule « dépliée » soit [0.6, 1.2). On ne calcule jamais [1, 1.2) : ce segment s'identifie à [0, 0.2), et ce segment-là est la cellule (coupée par D) de l'**image** y - 1 = -0.1. Le seed virtuel (-0.1, id = même i) apporte ce morceau. La cellule de i est donc { [0.6, 1) depuis y, [0, 0.2) depuis y - 1 }, et les deux morceaux tiennent dans D par construction.

En 2D, c'est pareil : on construit, pour chaque seed virtuel `( i, T )`, sa cellule de Laguerre **coupée par D** (la start cell), et rien d'autre. Une cellule de Laguerre classique coupée par D est toujours une bonne cellule, quel que soit le voisinage, *à condition que tous les seeds virtuels qui peuvent la couper soient présents*. Une cellule est donc fausse seulement s'il manque des images, jamais parce que la start cell est « trop petite ».

Et le manque d'images se règle **avant** de construire, pas après :

- `U` = la pire puissance dans D, bornée par excès par un diagramme bon marché, celui des seeds repliés seuls. Ajouter des seeds ne peut que la faire baisser.
- une image `( i, T )` est utile seulement si `dist( T y_i, D )² - w_i ≤ U` ;
- on génère toutes les images utiles, puis on construit une seule fois. Pas de recalcul.

En pratique : pour la boîte, U₀ demande un diagramme sans périodicité, ce qui coûte à peu près autant qu'un diagramme. Si on veut éviter ce surcoût, on peut prendre la valeur de l'itération de Newton précédente plus une marge, et ne vérifier qu'a posteriori (cas rare d'un recalcul). C'est un détail d'optimisation, ça ne change pas l'API.

➡️ **Recommended:** start cell = D exactement. Choisir les images *avant* la construction avec le critère `dist( T y_i, D )² - w_i ≤ U`. Démarrer avec U₀ exact, et remplacer ça par l'heuristique « U de l'itération précédente + vérification » seulement si un benchmark montre que ça coûte.

**Answer:**

Si je comprends bien, tu propose de calculer n * P cellules si P est le nombre d'images à considérer. Mais ça coutera beaucoup plus cher que de partir d'une start cell suffisamment grande et de chercher dans le BSP ce qui peut couper.
---

## Q2 : Décrire la périodicité par des recollements de facettes du domaine

Oui, ça a beaucoup de sens, et c'est même la description canonique d'un espace quotient : un polytope fondamental avec un **appariement de ses facettes** (théorème des polyèdres de Poincaré). Les isométries de recollement **engendrent** le groupe. C'est mieux que des générateurs libres :

- D est donné (le polytope de g), donc il n'y a plus de question « le support de g est-il un domaine fondamental ? ». L'ancien Q3 disparaît.
- Le sens est évident pour l'utilisateur : « la facette gauche est collée à la droite ».
- Les images utiles se trouvent naturellement : on traverse les facettes recollées.

Forme proposée :

```python
sdot.plan( f, g, periodicity = True )                   # box: opposite faces, every axis
sdot.plan( f, g, periodicity = [ True, False ] )        # box: per axis ( cylinder )
sdot.plan( f, g, periodicity = [ ( 0, 2 ), ( 1, 3 ) ] ) # facet pairings, indices into g's half-spaces
sdot.plan( f, g, periodicity = [ ( 0, 2, M ) ] )        # explicit linear part ( Klein bottle: a flip )
```

Pour un appariement `( a, b )`, l'isométrie envoie la facette a sur la facette b, en plaçant D de l'autre côté. Par défaut, c'est une translation, ce qui demande que les deux facettes soient parallèles, opposées, et de même forme. La linéaire `M` donnée en troisième élément couvre les rotations et les symétries. Si l'isométrie n'envoie pas a exactement sur b, c'est une erreur : il n'y a rien de sensé à calculer. Les indices de facettes sont ceux de `g.bounding_half_spaces()`.

Conséquence : on **abandonne** la forme « liste de générateurs libres » du round 1. Chaque générateur utile est un recollement, et `M` couvre les isométries non triviales.

➡️ **Recommended:** la périodicité se décrit par des recollements de facettes de D : `True`, une liste de booléens (boîte), ou une liste `( a, b [, M ] )`. Les générateurs libres sont abandonnés. On vérifie que chaque isométrie envoie a sur b.

**Answer:**

Ça m'a l'air très bien mais ça induit une perte de symétrie entre f et g. Je me demande s'il ne faudrait pas plutôt spécifier les coupes (direction + offset), avec évventuellement la possibilité d'utiliser quelque chose du genre `g.domain`.

---

## Q3 : Partiel déclenché par `target_mass` des distributions, sans kwarg sur `plan`

Ta remarque : « normalize » sert aussi à se ramener à du code connu. Et `target_mass` existe déjà dans `Image` (C++). Proposition :

- toutes les distributions prennent `target_mass = 1` par défaut, ou `None` pour garder la masse brute ;
- `plan` n'a **pas** de kwarg pour ça : les masses finales de f et g décident. Égales, c'est du transport classique. Différentes, c'est du partiel, dans le sens qui convient (round 2 Q7) ;
- « égales » veut dire : `target_mass` numériques identiques ⇒ égales exactement (cas par défaut, rien ne change). Sinon, la comparaison se fait en relatif à `tol` près.

➡️ **Recommended:** `target_mass` (1 par défaut, `None` = brut) sur toutes les distributions, dont `SumOfDiracs`. Partiel automatique selon les masses finales, pas de kwarg sur `plan`.

**Answer:**

ok
