# Ground metrics : round 2

> Leave **Answer** empty to accept the recommendation. `?` asks for more detail.
> Write anything else (answer, objection, comment) under any question, save, then say "ok" in the chat.

Décisions du round 1 : [decisions.md](decisions.md).

---

## Q1 : Start cell = domaine fondamental, sans agrandissement ; critère exact pour les images

Je pense que (ii) a été mal formulé au round 1. **La start cell n'a pas besoin d'être plus grande que le domaine.** On ne calcule pas des cellules « dépliées » dans ℝᵈ qui déborderaient. On calcule, *dans le domaine fondamental D seulement*, le diagramme de puissance de tous les seeds virtuels `( i, T )`. Un point x ∈ D appartient à la cellule de i si l'image de i la plus proche (au sens de la puissance) est une image de i. C'est exactement la définition de la cellule sur le tore. La cellule torique de i est donc l'union de ses morceaux `D ∩ Laguerre( T y_i )`, et chaque morceau tient dans D par construction.

La seule vraie question est **quelles images suffisent**. Pas besoin de partir d'un domaine infini ni de supposer « un anneau suffit ». On a un critère exact et peu coûteux :

- `U = max_{x ∈ D} min_{j,S} ( |x - S y_j|² - w_j )` : la pire puissance atteinte dans D. C'est le max, sur les sommets des cellules, de la puissance du propriétaire.
- L'image `( i, T )` ne peut posséder aucun point de D si `dist( T y_i, D )² - w_i > U`.
- Si on calcule U avec **un sous-ensemble** d'images (par exemple les seeds repliés seuls), on obtient une **borne supérieure** du vrai U, puisqu'ajouter des seeds ne peut que faire baisser le min. Le critère appliqué avec cette borne est donc conservateur et toujours juste.

Algorithme : (1) replier les seeds dans D (Q2). (2) Calculer le diagramme des seeds repliés seuls dans D, ce qui donne U₀ (et pour D une boîte, juste le max sur les sommets). (3) Générer toutes les images `( i, T )` telles que `dist( T y_i, D )² - w_i ≤ U₀`, en parcourant le groupe par anneaux jusqu'à ce qu'un anneau entier soit rejeté. (4) Calculer le diagramme final. Une seule passe, résultat garanti. Le poids w change à chaque itération de Newton : on refait (2) et (3) à chaque itération, ce qui ne coûte pas cher à côté du diagramme.

Ce critère remplace celui « en |t| > 2 · rayon » du round 1, qui était approximatif.

➡️ **Recommended:** start cell = D, exactement. Les images sont choisies par le critère `dist( T y_i, D )² - w_i ≤ U`, avec U borné par le diagramme des seeds repliés. Les images sont recalculées à chaque itération.

**Answer:**

Je ne suis pas sûr de comprendre. Si tu proposes de construire la cellule sur une start cell trop petite, on sera obligé de recalculer complètement la cellule si jamais ça dépasse, non ?

---

## Q2 : Repli des diracs dans le domaine fondamental, dépliage des résultats

C'est ta remarque. Si les diracs s'étalent sur plusieurs périodes, on commence par les replier : `y_i' = T_i y_i ∈ D`, avec T_i retenu pour chaque i. On résout sur les y_i', puis on déplie :

- barycentre : `b_i = T_i⁻¹ b_i'`, puis le dépliage par morceaux du round 1 (Q6 a) se fait autour de y_i' ;
- gradient par rapport aux positions : `∂/∂y_i = M_iᵀ ∂/∂y_i'`, une rotation pour les isométries non triviales ;
- les poids sont invariants.

Le domaine vient de g (Q3), pas des diracs. On ne cherche pas d'enveloppe des diracs.

➡️ **Recommended:** repli systématique en entrée (T_i stocké), dépliage systématique des sorties. C'est transparent pour l'utilisateur, ce qui rend compatible avec la décision « résultats dans le repère du seed ».

**Answer:**

Oui

---

## Q3 : Le support de g doit-il être un domaine fondamental ?

Avec `periodicity = True`, D est la boîte de g et c'est un domaine fondamental par construction. Avec des générateurs quelconques, il reste deux cas qui posent problème :

- supp g **plus grand** qu'un domaine fondamental : des zones sont comptées deux fois, et le résultat est faux ;
- supp g **plus petit** : g périodisée a des trous. C'est légitime, par exemple une densité nulle par endroits, mais alors D n'est plus le support de g.

Options : (a) D = boîte englobante de g, et on exige que ce soit un domaine fondamental. Pour les translations, on vérifie `vol( D ) = |det( réseau )|` et l'alignement ; sinon erreur. (b) D est déduit de la périodicité seule (parallélépipède des générateurs, ou domaine de Dirichlet), et g est repliée dans D (« chercher un domaine en se basant uniquement sur la périodicité », ta deuxième piste). Ça demande de savoir intégrer g repliée, donc de découper g par les translatés de D.

➡️ **Recommended:** (a) pour commencer. C'est le cas qui sert vraiment (textures, angles), et il ne demande rien de nouveau aux distributions. (b) va au roadmap.

**Answer:**

Je me demande si la formulation proposée pour la périodicité n'est pas à côté de la plaque.

Une solution pourrait par exemple être de préciser un polytope avec des correspondances entre les facettes des coupes. Est-ce que ça aurait du sens ?

---

## Q4 : Où vivent les seeds virtuels (CPU / GPU / BSP) ? (reprise de Q8)

Tu as raison, Q8 dépendait de Q7. Avec Q1, les seeds virtuels sont de **vrais seeds en plus**. Il y a deux implémentations possibles :

- (a) **Matérialiser** les seeds en amont : un tableau de positions `[ n + n_img, d ]`, des poids recopiés, et une table `virtual → i`. Le BSP, les noyaux CPU et GPU et `bisector` ne bougent pas. Il faut seulement que les sorties (masses, moments, dérivées de Newton) soient **réduites par id i**. Le surcoût mémoire est faible, puisque le critère de Q1 ne garde que les images proches du bord de D.
- (b) Générer les images à la volée dans les providers. On économise la mémoire, mais il faut toucher aux providers CPU, aux deux `.cuh` et au pruning.

La matrice de Newton demande attention dans les deux cas. Une facette entre `( i, T )` et `( j, S )` contribue au coefficient `( i, j )`, et peut-être à `( i, i )` si j = i : une cellule voisine de sa propre image. Ce cas ne se présente pas sans périodicité.

➡️ **Recommended:** (a). C'est le même code pour CPU et GPU, et la réduction par id sert aussi aux sorties. Le cas `j = i` (facette d'une cellule avec sa propre image) est traité explicitement dans l'assemblage du hessien, et testé. 1D + `periodicity` : erreur explicite (cercle = autre algorithme, au roadmap).

**Answer:**

On peut faire (a) pour l'heure.

---

## Q5 : Classement révisé (suite de ta réponse à Q1)

Tu ne passes à côté de rien : c'est mon axe 3 qui mélangeait deux choses. Dans le transport partiel, la boule de rayon √w_i ne vient pas d'une propriété du dirac. Elle vient d'un **dirac virtuel global** (le puits), de coût 0 et de poids 0 par invariance de jauge : `|x - y_i|² - w_i ≤ 0` donne la boule. C'est ton point de vue, et c'est le bon. Ce qui reste de « ce que le dirac peut capter » (un rayon de capture propre au dirac, par exemple `φ_i( r ) = r²` pour `r ≤ R_i`, sinon `+∞`) relève de l'axe φ, avec un φ qui dépend du dirac.

Le classement devient :

1. **Espace** : ℝᵈ ou ℝᵈ / G (périodicité).
2. **Coût φ** : global ou par dirac (fonctions radiales autour des diracs, rayon de capture…).
3. **Bilan de masse** : équilibré (cas actuel) ou non (puits global ⇒ transport partiel).

➡️ **Recommended:** adopter ce classement pour `docs/guide/ground_metrics.md` (espace / φ / bilan de masse).

**Answer:**

Oui

---

## Q6 : Comment déclencher le partiel : `normalize = False`

Aujourd'hui, f et g sont **toutes les deux normalisées** (`OtProblem.py:333`, `:352`). Déclencher le partiel dès que les masses diffèrent casserait le comportement actuel. Ça le déclencherait aussi à tort sur des écarts d'arrondi, par exemple une `Image` dont la somme vaut 0.9999.

Ta formulation « ne pas normaliser » règle ça proprement : `normalize = True` par défaut, ce qui ne change rien. Avec `normalize = False`, les masses brutes sont gardées, et si elles diffèrent (au-delà de `tol`), le transport devient partiel automatiquement, sans `partial = True` à préciser.

➡️ **Recommended:** un kwarg `normalize = True` sur `OtProblem` / `plan`. Avec `False`, on obtient le partiel automatique selon le côté excédentaire, et un écart sous `tol` est traité comme équilibré.

**Answer:**

Oui sur le fond, mais attention : la normalisation des densités dans sdot sert aussi à se rammener à des densités pour lesquelles on a du code. Il faudrait peut être préciser `normalize_masses` par exemple. Rq: dans les densités, l'approche est de faire un paramètre `target_mass` (ou autre nom si ce n'est pas claire) à 1 par défaut, qu'on peut mettre à None pour ne pas normaliser.

---

## Q7 : Les deux sens du partiel

Les deux sens sont mathématiquement différents :

- **mass( f ) < mass( g )**, avec trop de densité : le puits est du côté continu. Les cellules sont `Laguerre_i ∩ B( y_i, √w_i )`, il y a une seule nouvelle géométrie (l'intersection avec des boules), et le Newton est le même que d'habitude. C'est le cas « boules de rayon √w ».
- **mass( f ) > mass( g )**, avec trop de diracs : tout g est transporté, mais chaque dirac reçoit `m_i ≤ a_i`. Il n'y a pas de géométrie nouvelle (cellules de Laguerre ordinaires). Les contraintes deviennent des inégalités : les diracs pas pleins sont à `w_i` minimal commun. C'est un Newton avec contraintes de boîte ou d'activité.

➡️ **Recommended:** faire d'abord mass( f ) < mass( g ), puisque c'est lui qui demande l'intersection avec les boules et qui prépare l'axe φ. L'autre sens suit (c'est du solveur pur, sans géométrie). Avec `normalize = False`, d'ici là, mass( f ) > mass( g ) donne une erreur explicite.

**Answer:**

Très bien.

---

## Q8 : Variante pénalisée (coût de non-transport λ)

Il y a deux partiels classiques. (a) **Masse contrainte** : le puits prend exactement l'excédent, et son poids est une inconnue duale, fixée à 0 par jauge. C'est le cas de Q6/Q7. (b) **Pénalisé** : on transporte ce qui coûte moins que λ, et la masse transportée est libre. On fixe alors le poids du puits à `-λ`, et les boules ont pour rayon `√( w_i + λ )`. Le noyau géométrique est le même.

➡️ **Recommended:** prévoir (b) dans l'API (`unbalanced_cost = λ` ou un nom proche, à trancher au moment venu) mais ne l'implémenter qu'après (a). Ça ne change rien à la géométrie, c'est un réglage du solveur.

**Answer:**

Parfait