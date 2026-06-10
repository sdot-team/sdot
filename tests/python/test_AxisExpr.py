from sdot.aggregate.AxisExpr import AxisExpr


def _single_term( expr_str ):
    e = AxisExpr( expr_str )
    assert len( e.terms ) == 1
    return e.terms[ 0 ]


def test_constant():
    e = AxisExpr( "3" )
    assert e.offset == 3
    assert e.terms == []


def test_simple_var():
    t = _single_term( "dim" )
    assert t.coeff == 1
    assert t.name == "dim"
    assert t.selection is None
    assert t.arguments is None


def test_coeff_left():
    t = _single_term( "3 * dim" )
    assert t.coeff == 3
    assert t.name == "dim"


def test_coeff_right():
    t = _single_term( "dim * 3" )
    assert t.coeff == 3
    assert t.name == "dim"


def test_sum_with_offset():
    e = AxisExpr( "dim + 1" )
    assert e.offset == 1
    assert len( e.terms ) == 1
    assert e.terms[ 0 ].name == "dim"


def test_sum_two_vars():
    e = AxisExpr( "dim + nb_points" )
    assert e.offset == 0
    assert len( e.terms ) == 2
    names = { t.name for t in e.terms }
    assert names == { "dim", "nb_points" }


# --- [] notation (selection / dynamic tensor) --------------------------------

def test_subscript_single():
    # var[x]  →  selection = [ AxisExpr("x") ]
    t = _single_term( "nb_elements[ smurf ]" )
    assert t.name == "nb_elements"
    assert t.arguments is None
    assert isinstance( t.selection, list )
    assert len( t.selection ) == 1
    assert t.selection[ 0 ].terms[ 0 ].name == "smurf"


def test_subscript_multi():
    # var[x, y]  →  selection = [ AxisExpr("x"), AxisExpr("y") ]
    t = _single_term( "nb_elements[ smurf, dim ]" )
    assert t.name == "nb_elements"
    sel = t.selection
    assert isinstance( sel, list )
    assert len( sel ) == 2
    assert sel[ 0 ].terms[ 0 ].name == "smurf"
    assert sel[ 1 ].terms[ 0 ].name == "dim"


def test_subscript_empty():
    # var[]  →  selection = []  (not None)
    t = _single_term( "nb_elements[]" )
    assert t.name == "nb_elements"
    assert t.arguments is None
    assert t.selection == []


# --- () notation (expansion) -------------------------------------------------

def test_expansion_single():
    # var(x)  →  arguments = [ AxisExpr("x") ]
    t = _single_term( "nb_knots( dim )" )
    assert t.name == "nb_knots"
    assert t.selection is None
    assert isinstance( t.arguments, list )
    assert len( t.arguments ) == 1
    assert t.arguments[ 0 ].terms[ 0 ].name == "dim"


def test_expansion_in_sum():
    # nb_knots( dim ) + 1
    e = AxisExpr( "nb_knots( dim ) + 1" )
    assert e.offset == 1
    assert len( e.terms ) == 1
    v = e.terms[ 0 ]
    assert v.name == "nb_knots"
    assert v.arguments[ 0 ].terms[ 0 ].name == "dim"


def test_expansion_with_coeff():
    t = _single_term( "2 * nb_knots( dim )" )
    assert t.coeff == 2
    assert t.name == "nb_knots"
    assert t.arguments[ 0 ].terms[ 0 ].name == "dim"


# --- complex expression -------------------------------------------------------

def test_full_expression():
    e = AxisExpr( "3 * dim + nb_elements[ dim, nb_points ] + 1 + nb_items" )
    assert e.offset == 1
    assert len( e.terms ) == 3

    by_name = { t.name: t for t in e.terms }
    assert set( by_name ) == { "dim", "nb_elements", "nb_items" }

    assert by_name[ "dim" ].coeff == 3
    assert by_name[ "dim" ].selection is None

    sel = by_name[ "nb_elements" ].selection
    assert len( sel ) == 2
    assert sel[ 0 ].terms[ 0 ].name == "dim"
    assert sel[ 1 ].terms[ 0 ].name == "nb_points"

    assert by_name[ "nb_items" ].coeff == 1
    assert by_name[ "nb_items" ].selection is None
