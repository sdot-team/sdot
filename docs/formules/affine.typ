#set heading(numbering: "1.a")
#set page(fill: black)
#set text(fill: white)

= La grosse formule

$ W = sum_i integral_(c_i)^(c_(i+1)) ( x - p_i )^2 rho( x ) dif x space "avec" space c_i = Q( q_i ) $

où

$ q_i = frac( integral rho, sum w ) sum_( k < i ) w_k "(somme des poids normalisés)" $

qu'on pourra ecrire comme 

$ q_i = P r_i $

avec $ P = integral rho $ et $ r_i = frac( sum_( k < i ) w_k, sum w )  $

Rq: on pourrait ne pas vouloir normaliser les poids. Il faudra utiliser un flag pour le signaler.

= Position

$ frac( partial d, partial p_j ) = 2 integral_(c_j)^(c_(j+1)) ( p_j - x ) rho( x ) dif x $

En effet, $frac( partial c_j, partial p_i ) = 0$ puisque la position des cellules ne dépend pas de la position des diracs.

On a par conséquent

$ frac( partial d, partial p_i ) = 2 frac( P, sum w ) w_k ( p_i - B_i ) $
// & + 
// 

= Poids

$ frac( partial d, partial w_o ) = sum_i (
      frac( partial c_(i+1), partial w_o ) ( c_(i+1) - p_i )^2 rho( c_(i+1) ) 
    - frac( partial c_i, partial w_o ) ( c_i - p_i )^2 rho( c_i ) 
) $

Exemple à 3 diracs:

$ c_1 = Q( frac( P, w_0 + w_1 + w_2 ) w_0 )  = Q( P r_1 ) $

$ frac( c_1, partial w_0 ) = frac( 1, rho( q_1 ) ) frac( P, sum w ) ( 1 - r_1 ) $

$ frac( c_1, partial w_1 ) = frac( 1, rho( q_1 ) ) frac( P, sum w ) ( - r_1 ) $

$ frac( c_1, partial w_2 ) = frac( 1, rho( q_1 ) ) frac( P, sum w ) ( - r_1 ) $

De façon générale

$ frac( c_i, partial w_k ) = frac( 1, rho( q_1 ) ) frac( P, sum w ) ( 1_(k < i) - r_i ) $

Donc

$ frac( partial d, partial w_k ) = frac( P, sum w ) sum_i [
    ( c_(i+1) - p_i )^2 ( 1_(k < i+1) - r_(i+1) ) - ( c_i - p_i )^2 ( 1_(k < i) - r_i )
] $

== Avec 1 dirac

$ frac( partial d, partial w_0 )
 = frac( P, sum w ) [ ( c_1 - p_0 )^2 ( 1 - r_1 ) - ( c_0 - p_0 )^2 ( - r_0 ) ] 
 = 0
$

== Avec 3 diracs

=== Sans fixer le $k$

$ frac( partial d, partial w_k ) = frac( P, sum w ) 
    & [ ( c_1 - p_0 )^2 ( 1_(k < 1) - r_1 ) - ( c_0 - p_0 )^2 ( 1_(k < 0) - r_0 ) \
    & + ( c_2 - p_1 )^2 ( 1_(k < 2) - r_2 ) - ( c_1 - p_1 )^2 ( 1_(k < 1) - r_1 ) \
    & + ( c_3 - p_2 )^2 ( 1_(k < 3) - r_3 ) - ( c_2 - p_2 )^2 ( 1_(k < 2) - r_2 ) ] 
$

Ce qui se simplifie en 

$ frac( partial d, partial w_k ) = frac( P, sum w ) 
    [ & ( c_1 - p_0 )^2 ( 1_(k < 1) - r_1 ) \
    + & ( c_2 - p_1 )^2 ( 1_(k < 2) - r_2 ) - ( c_1 - p_1 )^2 ( 1_(k < 1) - r_1 ) \
    - & ( c_2 - p_2 )^2 ( 1_(k < 2) - r_2 ) ] 
$

C'est à dire 

$ frac( partial d, partial w_k ) = frac( P, sum w ) 
    [ & ( 1_(k < 1) - r_1 ) ( ( c_1 - p_0 )^2 - ( c_1 - p_1 )^2 ) \
    + & ( 1_(k < 2) - r_2 ) ( ( c_2 - p_1 )^2 - ( c_2 - p_2 )^2 )  ] 
$

=== Avec $k=0$

$ frac( partial d, partial w_k ) = frac( P, sum w ) 
    [ & ( 1 - r_1 ) ( ( c_1 - p_0 )^2 - ( c_1 - p_1 )^2 ) \
    + & ( 1 - r_2 ) ( ( c_2 - p_1 )^2 - ( c_2 - p_2 )^2 )  ] 
$

=== Avec $k=1$

$ frac( partial d, partial w_k ) = frac( P, sum w ) 
    [ & ( - r_1 ) ( ( c_1 - p_0 )^2 - ( c_1 - p_1 )^2 ) \
    + & ( 1 - r_2 ) ( ( c_2 - p_1 )^2 - ( c_2 - p_2 )^2 )  ] 
$
