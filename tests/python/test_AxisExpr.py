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
    assert t.variable.name == "dim"
    assert t.variable.selection is None
    assert t.variable.arguments is None


def test_coeff_left():
    t = _single_term( "3 * dim" )
    assert t.coeff == 3
    assert t.variable.name == "dim"


def test_coeff_right():
    t = _single_term( "dim * 3" )
    assert t.coeff == 3
    assert t.variable.name == "dim"


def test_sum_with_offset():
    e = AxisExpr( "dim + 1" )
    assert e.offset == 1
    assert len( e.terms ) == 1
    assert e.terms[ 0 ].variable.name == "dim"


def test_sum_two_vars():
    e = AxisExpr( "dim + nb_points" )
    assert e.offset == 0
    assert len( e.terms ) == 2
    names = { t.variable.name for t in e.terms }
    assert names == { "dim", "nb_points" }


# --- [] notation (selection / dynamic tensor) --------------------------------

def test_subscript_single():
    # var[x]  →  selection = [ AxisExpr("x") ]
    t = _single_term( "nb_elements[ smurf ]" )
    assert t.variable.name == "nb_elements"
    assert t.variable.arguments is None
    assert isinstance( t.variable.selection, list )
    assert len( t.variable.selection ) == 1
    assert t.variable.selection[ 0 ].terms[ 0 ].variable.name == "smurf"


def test_subscript_multi():
    # var[x, y]  →  selection = [ AxisExpr("x"), AxisExpr("y") ]
    t = _single_term( "nb_elements[ smurf, dim ]" )
    assert t.variable.name == "nb_elements"
    sel = t.variable.selection
    assert isinstance( sel, list )
    assert len( sel ) == 2
    assert sel[ 0 ].terms[ 0 ].variable.name == "smurf"
    assert sel[ 1 ].terms[ 0 ].variable.name == "dim"


def test_subscript_empty():
    # var[]  →  selection = []  (not None)
    t = _single_term( "nb_elements[]" )
    assert t.variable.name == "nb_elements"
    assert t.variable.arguments is None
    assert t.variable.selection == []


# --- () notation (expansion) -------------------------------------------------

def test_expansion_single():
    # var(x)  →  arguments = [ AxisExpr("x") ]
    t = _single_term( "nb_knots( dim )" )
    assert t.variable.name == "nb_knots"
    assert t.variable.selection is None
    assert isinstance( t.variable.arguments, list )
    assert len( t.variable.arguments ) == 1
    assert t.variable.arguments[ 0 ].terms[ 0 ].variable.name == "dim"


def test_expansion_in_sum():
    # nb_knots( dim ) + 1
    e = AxisExpr( "nb_knots( dim ) + 1" )
    assert e.offset == 1
    assert len( e.terms ) == 1
    v = e.terms[ 0 ].variable
    assert v.name == "nb_knots"
    assert v.arguments[ 0 ].terms[ 0 ].variable.name == "dim"


def test_expansion_with_coeff():
    t = _single_term( "2 * nb_knots( dim )" )
    assert t.coeff == 2
    assert t.variable.name == "nb_knots"
    assert t.variable.arguments[ 0 ].terms[ 0 ].variable.name == "dim"


# --- complex expression -------------------------------------------------------

def test_full_expression():
    e = AxisExpr( "3 * dim + nb_elements[ dim, nb_points ] + 1 + nb_items" )
    assert e.offset == 1
    assert len( e.terms ) == 3

    by_name = { t.variable.name: t for t in e.terms }
    assert set( by_name ) == { "dim", "nb_elements", "nb_items" }

    assert by_name[ "dim" ].coeff == 3
    assert by_name[ "dim" ].variable.selection is None

    sel = by_name[ "nb_elements" ].variable.selection
    assert len( sel ) == 2
    assert sel[ 0 ].terms[ 0 ].variable.name == "dim"
    assert sel[ 1 ].terms[ 0 ].variable.name == "nb_points"

    assert by_name[ "nb_items" ].coeff == 1
    assert by_name[ "nb_items" ].variable.selection is None
