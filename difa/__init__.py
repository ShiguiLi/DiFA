# SPDX-License-Identifier: MIT
"""DiFA: inference-time forward-process alignment."""
__version__ = '0.3.0'
__all__ = ['DiFA', 'Refinement', 'PredictionHistory', 'PredictionRecord', 'LogSNRMemory']


def __getattr__(name):
    # CPU packing/evaluation can import difa.io without loading PyTorch.
    from importlib import import_module
    modules = {'DiFA': 'core', 'Refinement': 'core',
               'PredictionHistory': 'history', 'PredictionRecord': 'history',
               'LogSNRMemory': 'memory'}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module('.' + modules[name], __name__), name)
    globals()[name] = value
    return value
