from typing import List

_optimize = True


def activate_optimize():
    global _optimize
    _optimize = True


def deactivate_optimize():
    global _optimize
    _optimize = False


if not _optimize:
    from numba.experimental import jitclass
    from numba import njit
    from numba.typed import List as numbaList
    from numba.core.types import ListType as numbaListType
    from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings

    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

else:

    def njit(f):
        return f


    def jitclass(*args, **kwargs):
        def decorated_class(original_class):
            class dummy:
                def __init__(dummy_self):
                    dummy_self.instance_type = original_class

            original_class.class_type = dummy()
            return original_class

        return decorated_class


    numbaList = lambda _list: _list
    numbaListType = lambda _type: List[_type]

