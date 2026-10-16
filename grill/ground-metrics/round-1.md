# Ground metrics : round 1

> Leave **Answer** empty to accept the recommendation. `?` asks for more detail.
> Write anything else (answer, objection, comment) under any question, save, then say "ok" in the chat.

État des lieux (vérifié dans le code) : aucune abstraction de coût aujourd'hui. `|x - y|² - w` est câblé partout : `bisector()` dans `cell/Plane.h`, recopié dans `gpu/Cell2D.cuh` et `gpu/Cell3D.cuh`, le pruning BSP, les moments, le noyau 1D, et le coût + gradient côté Python (`SdotPlanNd.py:1092`, `:1153`). `plan( f, g )` ne prend aucun paramètre. Le domaine vient de `g.bounding_half_spaces()`. `docs/guide/ground_metrics.md` tient en une ligne (et n'est pas suivi par git).

---

## Q1 : Le classement : trois axes indépendants plutôt qu'un objet « métrique »

Je pense que ce qui te manque est une distinction : « ground metric » mélange trois choses qui ne bougent pas au même endroit.

1. **L'espace** : ℝᵈ, ou ℝᵈ quotienté par un groupe G (tore, cylindre, groupes cristallographiques…). Le coût devient `c_G( x, y ) = min_{T ∈ G} c( x, T y )`. Ça marche pour *n'importe quel* c invariant par G. C'est une propriété du lieu où vivent f et g, pas de la fonction de coût.
2. **La fonction de coût** : `c( x, y ) = φ( |x - y| )`, avec φ(r) = r² par défaut (cellules polytopes, power diagram). Les autres φ donnent des bords courbes.
3. **Ce que le dirac peut capter** : une restriction radiale autour de chaque dirac. Le transport partiel en est le cas type : c'est le power diagram plus une option « non transporté » de coût 0. La cellule i devient `Laguerre_i ∩ B( y_i, √w_i )`. Ce n'est donc pas un nouveau coût. C'est un **puits** supplémentaire. Les fonctions radiales autour des diracs (ta remarque) vivent sur cet axe-là, ou sur l'axe 2 si φ dépend du dirac.

Conséquences : la périodicité n'a pas à attendre un objet `Metric`, et elle se compose plus tard avec φ et avec les boules sans rien changer.

➡️ **Recommended:** adopter ces trois axes (espace / φ / restriction radiale par dirac) comme classement, et le refléter dans `docs/guide/ground_metrics.md`.

**Answer:**

Le principe du classement me convient, mais il faut travailler sur "Ce que le dirac peut capter". Dans le cas du transport partiel, c'est comme si on avait un dirac virtuel global... ce qui est une façon différente de l'exprimer (et qui a priori ne donne qu'un façon de définir "Ce que le dirac peut capter"). La solution la plus basique pourrait être de dire "partial = True", ou de façon plus fondamental de dire qu'on ne veut pas normaliser les distributions. Si les distributions f et g n'ont pas la même masse, on est obligés de faire du partiel, et ce n'est pas la peine de le repréciser.

Est-ce que je passe à côté de quelque chose sur "Ce que le dirac peut capter" ?

---

## Q2 : Où se donne la périodicité ?

Options :

- (a) `sdot.plan( f, g, periodicity = ... )` : un kwarg de `plan`, `ot_solve` et `OtProblem`.
- (b) Sur la cible : `g = sdot.Image( ..., periodicity = ... )`, ou un wrapper `sdot.Periodic( g, ... )`. C'est `g` qui fournit déjà le domaine (`bounding_half_spaces`).
- (c) Un objet espace explicite : `sdot.plan( f, g, space = sdot.Torus( ... ) )`.

D'après Q1, la périodicité appartient à l'espace commun à f et g. (a) est le plus simple à écrire. (c) est plus propre conceptuellement mais ajoute un objet. (b) a l'avantage que le domaine fondamental *est* le support de g, donc les périodes par défaut sont évidentes.

➡️ **Recommended:** (a) un kwarg `periodicity` sur `OtProblem` (donc aussi `plan`, `ot_solve`, `w2`), stocké comme attribut du problème, comme `pb.source` et `pb.target`. Pour `PowerDiagram`, le même kwarg dans son constructeur. On garde la porte ouverte à (c) : si plus tard il y a un objet espace, `periodicity` en deviendra un raccourci.

**Answer:**

Parfait !

---

## Q3 : Quelles transformations accepter ?

Ton ancienne API acceptait toute transformation affine (matrice (d+1)², tuple `( M, V )`, ou vecteur). Mais le coût n'est invariant que si `M` est une **isométrie** : `|x - T y|² ≠ |T⁻¹ x - y|²` sinon, et le diagramme n'a plus de sens. Les isométries incluent les rotations et les symétries (bouteille de Klein, groupes cristallographiques, symétries miroir…).

➡️ **Recommended:** garder les trois formes d'entrée (vecteur, `( M, V )`, matrice (d+1)²), vérifier que `M` est orthogonale (à la tolérance près), sinon erreur. Implémenter d'abord le cas général via des images de seeds (rotation/symétrie comprises), puisque ça ne coûte pas plus cher qu'une translation dans `bisector`.

**Answer:**

Oui

---

## Q4 : Générateurs + clôture automatique, ou liste explicite des images ?

L'ancienne règle : chaque T ajouté entraîne `inv( T )`, `P @ T` et `P @ inv( T )` pour chaque P déjà présent. C'est une clôture à profondeur limitée. Pour `[1,0], [0,1]` elle donne les 8 voisins. Mais si une cellule dépasse une période (gros poids, diracs peu nombreux), il manque des images et le résultat est **faux en silence**.

➡️ **Recommended:** l'utilisateur donne des **générateurs**. La bibliothèque génère les images nécessaires : anneau 1 au départ, puis extension automatique quand une cellule touche le bord de la zone couverte par les images, au lieu de lever une erreur (« never raise when doable »). Le critère se calcule grâce au rayon de chaque cellule, qu'on connaît : une image de translation `t` est inutile si `|t| > 2 · max rayon`.

**Answer:**

Ok

---

## Q5 : Raccourcis pour le cas courant

Le cas courant, c'est `g` défini sur une boîte, périodique selon tous ses côtés.

➡️ **Recommended:** `periodicity = True` signifie : translations égales aux côtés de la boîte englobante de `g`, sur tous les axes. Une liste de booléens (`[ True, False ]`) donne la périodicité par axe (cylindre). Sinon, une liste de générateurs (Q3). Si `g` n'est pas une boîte alignée et que `True` est demandé : erreur claire.

**Answer:**

Très bien.

---

## Q6 : Repère des résultats (barycentres, positions, gradients)

La cellule de i peut être en plusieurs morceaux, chacun venant d'une image différente de y_i. Il faut choisir le repère pour les barycentres et pour `∂cost/∂positions`.

- (a) Tout est ramené dans le repère du seed i, avec des morceaux « dépliés » autour de y_i. Le barycentre peut alors sortir du domaine fondamental, mais `p ← bary` (Lloyd) reste correct et continu.
- (b) Barycentre replié dans le domaine fondamental. Il y a des sauts, et la moyenne de morceaux pris de part et d'autre est fausse.

➡️ **Recommended:** (a). Les positions peuvent être hors du domaine fondamental en entrée, sans repliement silencieux. Un helper `sdot.wrap( positions, periodicity )` est fourni pour qui veut replier.

**Answer:**

(a)

---

## Q7 : Densité cible : périodisée ou restreinte au domaine fondamental ?

Deux implémentations équivalentes. (i) On découpe les cellules dans ℝᵈ avec les images, puis on intègre g périodisée : g doit savoir s'évaluer hors de son support, ou on découpe la cellule par les translatés du domaine. (ii) On garde le domaine fondamental comme start cell, et les cuts viennent des images des *seeds*. Une cellule est alors l'union des morceaux `domaine ∩ Laguerre( T y_i )` sur toutes les images T. Ça colle à « domain = start cell » et ne demande rien de nouveau aux distributions.

➡️ **Recommended:** (ii). Le start cell reste le domaine fondamental. Chaque image d'un seed est un seed virtuel (même poids, position transformée, même id i), et l'intégration ne change pas. Ça passe par les providers (`ProviderAll`, `ProviderBsp`) : le BSP doit contenir les seeds virtuels ou savoir les générer.

**Answer:**

Oui pour les seed virtuels avec une seul start cell... Par contre, il y a un problème avec la "start cell" : elle doit être plus grande que le support du domaine. Pour être certains de ne pas faire de bêtises, on pourrait partir d'un domaine infini (on sait faire mais c'est lent). Est-ce qu'on pourrait faire mieux ? Sommes nous certains par exemple qu'une cellule pourra dépasser les bords initiaux agrandi 1 fois par les transformations de la périodicité (avec direct et inv à chaque fois bien sûr) ?

Rq: il y a un problème si les diracs s'étalent plus que la périodicité. Faudrait-il trouver un enveloppement des diracs puis replier avec la périodicité ? Ou alors faudrait-il chercher un domaine en se basant uniquement sur ce que donne la périodicité ?

Rq: si on repli, il faudra bien penser à déplier les barycentres.

---

## Q8 : Ordre CPU / GPU / 1D

Le GPU recopie `bisector` (`Cell2D.cuh`, `Cell3D.cuh`). Le 1D est un tout autre algorithme : le transport sur le cercle n'est pas un tri, c'est le problème de Delon–Salomon–Sobolevski.

➡️ **Recommended:** avec Q7 (ii), les seeds virtuels sont de simples seeds en plus, donc CPU et GPU en même temps et sans toucher à `bisector`. Pour le 1D périodique : à part, plus tard. D'ici là, `periodicity` en 1D donne une erreur explicite (c'est le seul cas où lever une erreur se justifie : pas d'algorithme).

**Answer:**

J'ai l'impression qu'on retombe sur Q7

---

## Q9 : Nommage de `w2` et `cost` une fois les métriques variables

`w2( f, g )` et `plan.cost` sont documentés comme W₂². Avec la périodicité, ça reste W₂² (sur le tore). Avec φ ≠ r², ou avec le partiel, ce n'est plus W₂.

➡️ **Recommended:** ne rien renommer maintenant. `w2` accepte `periodicity` puisque c'est encore W₂ sur l'espace quotient. La question reviendra avec l'axe φ, et on la tranchera à ce moment-là : probablement `sdot.cost( f, g, ... )` à côté de `w2`.

**Answer:**

Ok pour `cost`