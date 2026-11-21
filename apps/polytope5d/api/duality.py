"""Exact polar-dual metadata for the 69 analytically constructed 5D demos.

Conventions: P* = {u : <u,v> >= -1 for every v in P}.
This is deliberately NOT a generic polar hull algorithm for imported geometry.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations

from .db import demo_records


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


@lru_cache(maxsize=1)
def known_demos():
    return {obj["id"]: obj for obj in demo_records()}


def exact_dual(record: dict) -> dict | None:
    """Return polar and exact facet incidences only for trusted analytic examples."""
    if record.get("source") != "analytic_demo":
        return None
    expected = known_demos().get(record.get("id"))
    if expected is None or record.get("vertices_5d") != expected["vertices_5d"]:
        return None
    if record.get("edges") != expected["edges"]:
        return None

    name = record["id"]
    primal = record["vertices_5d"]
    if name.startswith("demo-simplex-"):
        dual = expected["facet_normals"]
        dual_edges = [list(pair) for pair in combinations(range(6), 2)]
    elif name == "demo-hypercube-5d":
        other = known_demos()["demo-crosspolytope-5d"]
        dual, dual_edges = other["vertices_5d"], other["edges"]
    elif name == "demo-crosspolytope-5d":
        other = known_demos()["demo-hypercube-5d"]
        dual, dual_edges = other["vertices_5d"], other["edges"]
    else:
        return None

    # Incidence for supporting hyperplanes <u,v> = -1. Both graphs are exact.
    facets = [[i for i, vertex in enumerate(primal) if dot(normal, vertex) == -1]
              for normal in dual]
    dual_facets = [[i for i, normal in enumerate(dual) if dot(vertex, normal) == -1]
                   for vertex in primal]
    if any(dot(u, v) < -1 for u in dual for v in primal):
        raise ValueError("Nonpolar analytic reference data")
    expected_vertices = {"demo-hypercube-5d": 16, "demo-crosspolytope-5d": 5}
    face_size = expected_vertices.get(name, 5)
    if any(len(f) != face_size for f in facets):
        raise ValueError("Invalid facet incidence in analytic reference")

    return {
        "dual_available": True,
        "dual_method": "exact_analytic_demo",
        "facets": facets,
        "facet_normals": dual,
        "dual_vertices_5d": dual,
        "dual_edges": dual_edges,
        "dual_facets": dual_facets,
        "dual_facet_normals": primal,
        "dual_vertex_count": len(dual),
        "dual_facet_count": len(dual_facets),
    }


def enrich_detail(record: dict) -> dict:
    """Add polar data without changing the persisted SQLite schema."""
    polar = exact_dual(record)
    if polar is None:
        return {**record, "dual_available": False, "dual_method": None,
                "dual_vertices_5d": None, "dual_edges": None,
                "dual_facets": None, "dual_facet_normals": None,
                "dual_vertex_count": None, "dual_facet_count": None,
                "facets": None}
    return {**record, **polar}
