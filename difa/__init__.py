"""Generic DiFA refinement and reference construction, with optional adapters."""
from .core import DiFA,Refinement
from .reference import SpatialReference
from .ops import joint_reference

__all__=['DiFA','Refinement','SpatialReference','joint_reference','SiTG03Guidance','G03Guidance']


def __getattr__(name):
    # Importing the universal tensor algorithm does not import SiT adapters.
    if name in {'SiTG03Guidance','G03Guidance'}:
        from .adapters.sit import SiTG03Guidance
        return SiTG03Guidance
    raise AttributeError(name)
