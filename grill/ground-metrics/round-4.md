# Ground metrics : round 4

> Leave **Answer** empty to accept the recommendation. `?` asks for more detail.
> Write anything else (answer, objection, comment) under any question, save, then say "ok" in the chat.

---

## Q1 : Une cellule par vrai seed, avec une start cell assez grande *et prouvée suffisante*

Tu as raison sur le coût. Une précision tout de même : avec le critère de U, les images retenues ne sont pas n·P. Ce sont seulement les seeds proches du bord de D, en O( n^((d-1)/d) ). Mais ta voie est meilleure : **n cellules, une par vrai seed**, construites dans une start cell assez grande. Les images ne servent alors que de *coupes* trouvées par le BSP.

Bonne nouvelle : on a une taille de start cell **exacte et indépendante des poids**. La cellule de i ne peut pas dépasser la cellule de Voronoï de y_i parmi **ses propres images** : y_i et T y_i ont le même poids, donc leur bissectrice de puissance est la médiatrice ordinaire. La start cell de i est donc la cellule de Dirichlet du groupe centrée en y_i. Pour une boîte torique, c'est simplement **D translatée pour être centrée sur y_i**. C'est la réponse au « sommes-nous certains » du round 1 : oui, et sans domaine infini.

Le reste vient de cette start cell :

- **Coupes** : le BSP des vrais seeds est interrogé avec la cellule translatée par `T⁻¹`, pour chaque T dont l'image de D touche la start cell. Pour la boîte, ce sont les 3ᵈ - 1 voisins, plus l'identité. Aucune copie de seeds, et le pruning existant fonctionne tel quel (requête translatée). C'est la variante (b) du round 2 Q4 (images à la volée), et elle devient plus simple que (a). Ça remet en cause « (a) pour l'heure ».
- **Intégration** : la cellule (dépliée, centrée sur y_i) recouvre au plus 2ᵈ translatés de D. On la coupe par les plans des facettes de D. Chaque morceau est ramené dans D par son T⁻¹ et intégré contre g. Barycentres et moments sont accumulés dans le repère de y_i, ce qui donne directement le « déplié » décidé au round 1 (Q6 a).
- **Hessien** : une facette entre i et `S y_j` contribue à `( i, j )`, comme d'habitude. Le cas `j = i` (facette avec sa propre image) est le seul cas nouveau.

➡️ **Recommended:** une cellule par vrai seed. La start cell est la cellule de Dirichlet du groupe centrée sur le seed (pour la boîte : D recentrée). Les coupes viennent du BSP interrogé par translation (pas de seeds matérialisés). L'intégration se fait par découpage selon les facettes de D, puis repli de chaque morceau. On remplace la décision « (a) pour l'heure » par cette approche.

**Answer:**

ok

---

## Q2 : Décrire le groupe, pas un domaine (symétrie f / g)

Avec Q1, l'algorithme n'a besoin que du **groupe** (les isométries) : la start cell s'en déduit. Le domaine sert uniquement à replier g, qui a déjà son support. On peut donc décrire la périodicité sans rien emprunter à g, ce qui rétablit la symétrie que tu cherches.

Options :

- (a) **Générateurs du groupe**, comme dans ton ancienne API : vecteurs (translations) ou `( M, V )`. C'est symétrique en f et g. Le support de g doit tiler par le groupe, ce qu'on vérifie : volume égal à |det| pour les translations, et facettes appariées par le groupe.
- (b) **Coupes** : paires `( direction, offset_min, offset_max )`, c'est-à-dire une tranche qui donne la translation `direction · ( offset_max - offset_min ) / |direction|²`. C'est aussi symétrique. Les offsets fixent *en plus* un domaine D, qui ne sert qu'à vérifier ou à replier g. Avec `periodicity = g.domain`, on apparie les facettes parallèles opposées de g.
- Dans les deux cas : `True` comme raccourci pour `g.domain` (boîte, tous les axes), et `[ True, False ]` par axe.

(b) dit un peu plus que (a), puisque les offsets sont redondants avec le support de g. Sa seule vraie utilité serait un D différent du support de g, par exemple une densité à support plus petit que la période, avec des trous. Mais (a) gère ce cas aussi : une densité nulle sur une partie de son support, ou un support plus petit qui tile avec des « trous » de masse nulle. Il suffit de relâcher la vérification en inclusion dans un domaine fondamental.

➡️ **Recommended:** (a). Générateurs (vecteurs ou `( M, V )`, `M` orthogonale), avec clôture et choix des images par l'algorithme. Raccourcis `True` / `[ True, False ]` / `g.domain` (ce dernier convertit les facettes opposées en translations). Vérification : le support de g est contenu dans un domaine fondamental du groupe.

**Answer:**

ok