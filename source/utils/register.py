from functools import wraps


"""
   Defining register to store building blocks. This is later used by the argument parser.
"""
Registers = {
    "LOSSES"          : dict(),
    "MODULES"         : dict(),
    "MODELS"          : dict(),
    "DATASETS"        : dict(),
    "TASKS"           : dict(),
    "CALLBACKS"       : dict(),
    "METRICS"         : dict()
}


def register(type : str) :
    """
    Decorator used to store a given elements in its register
    :param type: register to store the element in
    """
    if type not in Registers.keys() :
        raise ValueError(f"Unsupported type : {type}. Available type : {list(Registers)}")

    def _register(cls) :
        @wraps(cls)
        def _register_elt(cls) :
            for name in getattr(cls, "_names", []) + [cls.__name__] :
                Registers[type][name] = cls
            return cls
        return _register_elt(cls)
    return _register


