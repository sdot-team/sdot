#set heading(numbering: "1.a")
#set page(fill: black)
#set text(fill: white)

= La grosse formule

    $ W = sum_i integral_(Q_i)^(Q_(i+1)) ( x - p_i )^2 rho( x ) dif x $

= grad w en integrant la normalisation

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


= grad ys

    $ W = sum_i integral_(Q_i)^(Q_(i+1)) ( x - p_i )^2 rho( x ) dif x $

    $ ( partial W ) / ( partial y_z ) = \
        sum_i integral_(Q_i)^(Q_(i+1)) ( x - p_i )^2 rho'( x ) dif x \

        + sum_i [ ( partial Q_(i+1) ) / ( partial w_z ) ( Q_(i+1) - p_i )^2 rho( Q_(i+1) ) ] \

        - sum_i [ ( partial Q_(i  ) ) / ( partial w_z ) ( Q_(i  ) - p_i )^2 rho( Q_i ) ]
    $

    avec $ rho = rho^r / (integral rho^r ) $

