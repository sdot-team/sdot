#set heading(numbering: "1.a")
#set page(fill: black)
#set text(fill: white)

= La grosse formule

    $ W = frac( 1, integral rho ) sum_i integral_(c_i)^(c_(i+1)) ( x - p_i )^2 rho( x ) dif x $

avec 
    $ space c_i = Q( q_i ) $

où
    $ q_i = frac( sum_( k < i ) w_k, sum w )  $

Les poids *et* la densité sont normalisés par défaut. Il faudra prévoir des flags et adapté les codes pour le cas contraire.

= Position

$ frac( partial d, partial p_j ) = frac( 2, integral rho ) integral_(c_j)^(c_(j+1)) ( p_j - x ) rho( x ) dif x $

En effet, $frac( partial c_j, partial p_i ) = 0$ puisque la position des cellules ne dépend pas de la position des diracs.

On a par conséquent

$ frac( partial d, partial p_i ) = 2 frac( w_k, sum w ) ( p_i - B_i ) $
// & + 
// 

= Poids

$ frac( partial d, partial w_o ) = frac( 1, integral rho ) sum_i (
      frac( partial c_(i+1), partial w_o ) ( c_(i+1) - p_i )^2 rho( c_(i+1) ) 
    - frac( partial c_i, partial w_o ) ( c_i - p_i )^2 rho( c_i ) 
) $

Exemple à 3 diracs:

$ c_1 = Q( frac( w_0, w_0 + w_1 + w_2 ) ) = Q( q_1 ) $

$ frac( c_1, partial w_0 ) = frac( integral rho, rho( q_1 ) ) frac( 1 - q_1, sum w ) $

$ frac( c_1, partial w_1 ) = frac( integral rho, rho( q_1 ) ) frac( - q_1, sum w ) $

$ frac( c_1, partial w_2 ) = frac( integral rho, rho( q_1 ) ) frac( - q_1, sum w ) $

De façon générale

$ frac( c_i, partial w_k ) = frac( integral rho, rho( q_1 ) ) frac( 1_(k < i) - q_i, sum w ) $

Donc

$ frac( partial d, partial w_k ) = frac( 1, sum w ) sum_i [
    ( c_(i+1) - p_i )^2 ( 1_(k < i+1) - q_(i+1) ) - ( c_i - p_i )^2 ( 1_(k < i) - q_i )
] $

== Avec 1 dirac

$ frac( partial d, partial w_0 )
 = frac( 1, sum w ) [ ( c_1 - p_0 )^2 ( 1 - q_1 ) - ( c_0 - p_0 )^2 ( - q_0 ) ] 
 = 0
$

== Avec 3 diracs

=== Sans fixer le $k$

$ frac( partial d, partial w_k ) = frac( 1, sum w ) 
    & [ ( c_1 - p_0 )^2 ( 1_(k < 1) - q_1 ) - ( c_0 - p_0 )^2 ( 1_(k < 0) - q_0 ) \
    & + ( c_2 - p_1 )^2 ( 1_(k < 2) - q_2 ) - ( c_1 - p_1 )^2 ( 1_(k < 1) - q_1 ) \
    & + ( c_3 - p_2 )^2 ( 1_(k < 3) - q_3 ) - ( c_2 - p_2 )^2 ( 1_(k < 2) - q_2 ) ] 
$

Ce qui se simplifie en 

$ frac( partial d, partial w_k ) = frac( 1, sum w ) 
    [ & ( c_1 - p_0 )^2 ( 1_(k < 1) - q_1 ) \
    + & ( c_2 - p_1 )^2 ( 1_(k < 2) - q_2 ) - ( c_1 - p_1 )^2 ( 1_(k < 1) - q_1 ) \
    - & ( c_2 - p_2 )^2 ( 1_(k < 2) - q_2 ) ] 
$

C'est à dire 

$ frac( partial d, partial w_k ) = frac( 1, sum w ) 
    [ & ( 1_(k < 1) - q_1 ) ( ( c_1 - p_0 )^2 - ( c_1 - p_1 )^2 ) \
    + & ( 1_(k < 2) - q_2 ) ( ( c_2 - p_1 )^2 - ( c_2 - p_2 )^2 )  ] 
$

=== Avec $k=0$

$ frac( partial d, partial w_k ) = frac( 1, sum w ) 
    [ & ( 1 - q_1 ) ( ( c_1 - p_0 )^2 - ( c_1 - p_1 )^2 ) \
    + & ( 1 - q_2 ) ( ( c_2 - p_1 )^2 - ( c_2 - p_2 )^2 )  ] 
$

=== Avec $k=1$

$ frac( partial d, partial w_k ) = frac( 1, sum w ) 
    [ ( - q_1 )   & ( ( c_1 - p_0 )^2 - ( c_1 - p_1 )^2 ) \
    + ( 1 - q_2 ) & ( ( c_2 - p_1 )^2 - ( c_2 - p_2 )^2 )  ] 
$

= Ys de la fonction affine

$ W = frac( 1, integral rho ) sum_i integral_(c_i)^(c_(i+1)) ( x - p_i )^2 rho( x ) dif x space "avec" space c_i = Q( q_i ) $

$ frac( partial W, partial y_k ) = A + B +C $

avec 

$ A & = - frac( W_2^2, ( integral rho ) ^ 2 ) integral frac( partial rho, partial y_k ) \
  B & = frac( 1, integral rho ) sum_i integral_(c_i)^(c_(i+1)) ( x - p_i )^2 frac( partial rho, partial y_k )( x ) dif x  \
  C & = frac( 1, integral rho ) sum_i [ Q'( q_(i+1) ) ( c_(i+1) - p_i )^2 rho( c_(i+1) ) - Q'( c_i ) ( c_i - p_i )^2 rho( c_i ) ] \
    & = frac( 1, integral rho ) sum_i [ 
      ( integral_( -infinity )^( q_(i+1) ) frac( partial rho, partial y_k ) ) ( c_(i+1) - p_i )^2 - 
      ( integral_( -infinity )^( q_i ) frac( partial rho, partial y_k ) ) ( c_i - p_i )^2 ] \
  $

