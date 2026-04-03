#set heading(numbering: "1.a")
#set page(fill: black)
#set text(fill: white)

= Normalisation

= La grosse formule

    $ W = sum_i integral_(Q_i)^(Q_(i+1)) ( x - p_i )^2 rho( x ) dif x $

= On essaye en integrant la normalisation

    $P$ selon l'axe des primitives

    $ P_i = ( sum_(k<i) w_k ) / ( sum_l w_l ) ", " S = sum_l w_l ", " A_i = sum_(l<i) w_l $

    $ ( partial P_i ) / ( partial w_z ) =
        & ( 1 - A_i ) / S^2 "si " i > z \
        & - A_i / S^2 "si " i <= z
    $

    Or

    $ ( partial W ) / ( partial w_z ) = sum_i [ ( partial P_(i+1) ) / ( partial w_z ) ( Q_(i+1) - p_i )^2 ]
                                      - sum_i [ ( partial P_(i  ) ) / ( partial w_z ) ( Q_(i  ) - p_i )^2 ]
    ] $

    Donc

    $ ( partial W ) / ( partial w_z ) =
        & - sum_( i+1 <= z ) [ A_(i+1) / S^2 ( Q_(i+1) - p_i )^2 ] \
        & + sum_( i+1 >  z ) [ ( S - A_(i+1) ) / S^2 ( Q_(i+1) - p_i )^2 ] \

        & + sum_( i   <= z ) [ A_i / S^2 ( Q_i - p_i )^2 ] \
        & - sum_( i   >  z ) [ ( S - A_i ) / S^2 ( Q_i - p_i )^2 ]
    ] $

    $ S^2 ( partial W ) / ( partial w_z ) =
        & - sum_( i+1 <= z ) [ A_(i+1) ( Q_(i+1) - p_i )^2 ] \
        & + sum_( i+1 >  z ) [ ( S - A_(i+1) ) ( Q_(i+1) - p_i )^2 ] \

        & - sum_( i   <= z ) [ A_i ( Q_i - p_i )^2 ] \
        & - sum_( i   >  z ) [ ( S - A_i ) ( Q_i - p_i )^2 ]
    ] $



=== Ex avec 3 diracs:

    $ P_0 &= 0 \
      P_1 &= w_0 / ( w_0 + w_1 + w_2 ) \
      P_2 &= ( w_0 + w_1 ) / ( w_0 + w_1 + w_2 ) \
      P_3 &= 1 $

Donc
    $ ( partial P_0 ) / ( partial w_j ) = 0 $

    $ ( partial P_1 ) / ( partial w_0 ) &= ( s - w_0 ) / s^2 \
      ( partial P_1 ) / ( partial w_1 ) &= - w_0 / s^2 \
      ( partial P_1 ) / ( partial w_2 ) &= - w_0 / s^2
    $

    $ ( partial P_2 ) / ( partial w_0 ) &= ( s - w_0 - w_1 ) / s^2 \
      ( partial P_2 ) / ( partial w_1 ) &= ( s - w_0 - w_1 ) / s^2 \
      ( partial P_2 ) / ( partial w_2 ) &= (   - w_0 - w_1 ) / s^2
    $

    $ ( partial P_3 ) / ( partial w_j ) = 0 $

=== Ex avec 2 diracs:

    $ W =
        integral_(Q_0)^(Q_1) ( x - p_0 )^2 rho( x ) dif x +
        integral_(Q_1)^(Q_2) ( x - p_1 )^2 rho( x ) dif x +
        integral_(Q_2)^(Q_3) ( x - p_2 )^2 rho( x ) dif x
    $

    $ ( partial W ) / ( partial w_0 )
            & = ( s - w_0 ) / s^2 * ( Q_1 - p_0 )^2 \ // Q1
            // Q0
            & + ( s - w_0 - w_1 ) / s^2 * ( Q_2 - p_1 )^2 \ // Q2
            & - ( s - w_0 ) / s^2 * ( Q_1 - p_1 )^2 \ // Q1
            // Q3
            & - ( s - w_0 - w_1 ) / s^2 * ( Q_2 - p_2 )^2 \ // Q2
    $

    $ ( partial W ) / ( partial w_0 )
            & = ( s - w_0 ) / s^2 * ( Q_1 - p_0 )^2 \ // Q1
            // Q0
            & + ( s - w_0 - w_1 ) / s^2 * ( Q_2 - p_1 )^2 \ // Q2
            & - ( s - w_0 ) / s^2 * ( Q_1 - p_1 )^2 \ // Q1
            // Q3
            & - ( s - w_0 - w_1 ) / s^2 * ( Q_2 - p_2 )^2 \ // Q2
     $
