# SPDX-License-Identifier: MIT
"""Raw-observation memory helpers for DiFA adapters."""
from ._impl.guidance import LogSNRMemory
from ._impl.components import component_statistics, standardize_components

def response_components(response):
    return tuple(part for part, _, _ in component_statistics(response))

__all__ = ['LogSNRMemory', 'standardize_components', 'response_components']
