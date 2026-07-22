"""
PATHFINDER — Provably Near-Optimal Multi-hop Knowledge Retrieval
via Submodular Coverage Maximization on Multidimensional Knowledge Graphs.

Python package implementing Algorithm 1 from the PATHFINDER paper.
"""

from pathfinder.graph import KnowledgeGraph, KGBuilder
from pathfinder.algorithm import (
    PathfinderGreedy,
    TraversalResult,
    compute_F,
    compute_sigma,
    marginal_gain,
)
from pathfinder.facets import NodeFacets

__version__ = "0.1.0"
__all__ = [
    "KnowledgeGraph",
    "KGBuilder",
    "PathfinderGreedy",
    "TraversalResult",
    "compute_F",
    "compute_sigma",
    "marginal_gain",
    "NodeFacets",
]
