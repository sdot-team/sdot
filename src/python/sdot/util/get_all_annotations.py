import inspect

def get_all_annotations( cls ) -> dict:
    result = {}
    for base in reversed( cls.__mro__ ):
        result.update( inspect.get_annotations( base ) ) #
    return result
