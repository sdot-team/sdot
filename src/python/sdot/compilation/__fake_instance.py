# from .collect_attributes import collect_attributes


# def fake_instance( driver, return_type, *type_args, **type_kwargs ):
#     if callable( getattr( return_type, "fake_instance_for", None ) ):
#         return return_type.fake_instance_for( driver, *type_args, **type_kwargs )

#     if return_type is float:
#         raise NotImplementedError

#     if return_type is int:
#         raise NotImplementedError

#     attrs = {}
#     for name, t in collect_attributes( return_type ):
#         attrs[ name ] = fake_instance( driver, t, *type_args, **type_kwargs )
#     return return_type( **attrs )
