from ..drivers.driver import driver
from .IoCategory import IoCategory

from typing import Optional
from weakref import ref

class CallArg:
    """
        Recursive analysis of an argument sent to driver.call( ... )

        CallArg is an abstract base class.

        Allows for
        - re-assembly and update of outputs of the ffi calls (which are basically flat lists of tensors)
        - generation of the assembly code for the C++ side
    """

    io_category    : IoCategory

    python_value   : Optional[ any ]
    python_class   : any

    sub_dict       : dict[ str, 'CallArg' ]

    name_in_parent : Optional[ str ]
    orig_parent    : Optional[ ref ]
    parent         : Optional[ ref ]

    ctor_kwargs    : dict
    ctor_args      : list

    def __init__(
        self,
        name_in_parent : Optional[ str ],
        parent         : Optional[ ref ],
        python_class   : any,
        python_value   : Optional[ any ],
        io_category    : IoCategory,
        ctor_args      : Optional[ list ] = None,
        ctor_kwargs    : Optional[ dict ] = None
    ):
        self.name_in_parent = name_in_parent
        self.orig_parent    = None
        self.parent         = ref( parent ) if parent is not None else None
        self.python_value   = python_value
        self.python_class   = python_class
        self.sub_dict       = {}
        self.io_category    = io_category
        self.ctor_kwargs    = ctor_kwargs if ctor_kwargs is not None else {}
        self.ctor_args      = ctor_args if ctor_args is not None else []

    @staticmethod
    def factory( call_args, parent, name_in_parent, python_class, python_value, io_category: IoCategory, ctor_args, ctor_kwargs ):
        # value method
        if python_value is not None and callable( getattr( python_value, "call_arg_factory", None ) ):
            return python_value.call_arg_factory( call_args, parent, name_in_parent, io_category, ctor_args, ctor_kwargs )

        # class method. Used for instance for Tensor() for which `value` is an array
        if callable( getattr( python_class, "call_arg_factory", None ) ):
            return python_class.call_arg_factory( call_args, parent, name_in_parent, python_value, io_category, ctor_args, ctor_kwargs )

        # arrays
        if driver.is_a_tensor( python_value ):
            from .CallArg_Tensor import CallArg_Tensor
            return CallArg_Tensor.factory( call_args, parent, name_in_parent, python_class, python_value, io_category, ctor_args, ctor_kwargs, comes_from_basic_array = True )

        # std objects
        if isinstance( python_value, ( int, float ) ):
            if io_category.want_output:
                raise NotImplementedError # A tensor returning item() ?
            from .CallArg_Parameter import CallArg_Parameter
            return CallArg_Parameter.factory( call_args, name_in_parent, python_value )

        if isinstance( python_value, ( list, tuple ) ):
            raise NotImplementedError

        if python_class.__name__ == "NoneType":
            raise RuntimeError( f"for { name_in_parent }" )

        # else, get attributes
        from .CallArg_Aggregate import CallArg_Aggregate
        return CallArg_Aggregate.factory( call_args, parent, name_in_parent, python_class, python_value, io_category, ctor_args, ctor_kwargs )

    def signature( self ) -> str:
        raise NotImplementedError

    def get_template_args( self, template_args, names ):
        # by defaut: look in children
        for name, attr in self.sub_dict.items():
            attr.get_template_args( template_args, names + [ name ] )

    def cpp_type_name( self, main_list ) -> str:
        raise NotImplementedError

    def backward_version( self, call_args, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs = None ) -> 'CallArg':
        raise NotImplementedError

    def generate_structures( self, already_visited ):
        pass

    def get_includes( self, includes: set ):
        pass

    def get_ct_axis_variable_names( self, ct_axis_variable_names: list[ str ], name_list_so_far: list[ str ] ):
        # by defaut: look in children
        for name, arg in self.sub_dict.items():
            arg.get_ct_axis_variable_names( ct_axis_variable_names, name_list_so_far + [ name ] )

    def get_axis_variable_names( self, axis_variable_names: list[ str ] ):
        pass

    def get_arg_decl( self, non_differentiable_inputs: list, differentiable_inputs: list, parameters: list, outputs: list ):
        pass

    def add_axis_tensor_sources( self, sources, attributes, use_attributes, recursive ):
        # by default a node contributes no tensor source (tensors and aggregates override)
        pass

    def check_axis_consistency( self ):
        # by default: check each child (aggregates also check their own scope)
        for arg in self.sub_dict.values():
            arg.check_axis_consistency()

    def value_of_axis_variable( self, variable_name: str, is_a_dyn_size: bool = False ) -> int:
        res = self._find_axis_variable( variable_name, [ variable_name ], is_a_dyn_size, came_from = None )
        if res is None:
            raise RuntimeError( f"Unable to find axis variable value for '{ variable_name }' in '{ self.name_in_parent }'" )
        return res

    def _find_axis_variable( self, name: str, candidates: list[ str ], is_a_dyn_size: bool, came_from ) -> Optional[ int ]:
        """
        Resolve an axis variable from this node, in three directions:
          - locally, via explicit `ctor_kwargs` and the local shape system;
          - descending into child aggregates (which form their own scopes);
          - ascending to parents to find an explicit `ctor_kwargs`, where the name is
            tried both bare and prefixed by the path of crossed children
            (`dim` -> `cell_dim` -> `powerdiagram_cell_dim`).
        `came_from` is the neighbour we arrived from, to avoid bouncing back.
        Returns None when unresolvable here.
        """
        # 1. explicit value given through ctor_kwargs (bare or prefixed candidate names)
        for candidate in candidates:
            if candidate in self.ctor_kwargs:
                return int( self.ctor_kwargs[ candidate ] )

        # 2. a child standing for a dynamic axis carries its value directly
        child = self.sub_dict.get( name )
        if child is not None and child.python_value is not None:
            val = child.python_value
            return int( val.item() ) if hasattr( val, "item" ) else int( val )

        # 3. local linear system over the shapes of the directly contained tensors
        if axis_system := getattr( self, "axis_system", None ):
            value = axis_system( recursive = False ).local_value_of( name )
            if value is not None:
                return value

        # 4. descend into children (the bare name applies inside their scope). This also
        # reaches an explicit value attached to a leaf tensor, e.g. Return( Tensor, dim=2 ).
        for sub in self.sub_dict.values():
            if sub is came_from:
                continue
            value = sub._find_axis_variable( name, [ name ], is_a_dyn_size, came_from = self )
            if value is not None:
                return value

        # 5. ascend, accumulating this node's name as a prefix for parent ctor_kwargs
        for parent_ref in ( self.orig_parent, self.parent ):
            if parent_ref is None:
                continue
            p = parent_ref()
            if p is None or p is came_from:
                continue
            up_candidates = list( candidates )
            if self.name_in_parent:
                up_candidates += [ f"{ self.name_in_parent }_{ c }" for c in candidates ]
            value = p._find_axis_variable( name, up_candidates, is_a_dyn_size, came_from = self )
            if value is not None:
                return value

        return None

    def beg_with_same_shape( self, name, s, lines ):
        lines.append( s + f"{ name }.with_same_shape( [&]( auto &{ name } ) {{" )
        return s + "  "

    def end_with_same_shape( self, name, s, lines ):
        s = s[ :-2 ]
        lines.append( s + "} );" )
        return s

    def assembled_code( self, beg_line: str ) -> str:
        raise NotImplementedError

    def assemble_return( self ) -> any:
        raise NotImplementedError

    def fully_qualified_name( self ) -> str:
        l = [ self.name_in_parent or "" ]
        p = self.parent
        while p is not None:
            if name := p().name_in_parent:
                l.append( name )
            p = p().parent
        l.reverse()
        return str.join( "_", l[ 1: ] )
