"""Exact 3D facet/edge/polar construction for convex reflexive lattice polytopes.

We use the sign convention P* = {u | <u,v> >= -1 for all v in P}.
All supporting equations are validated exactly using Python integers.
Floating point is used only to cyclically ORDER vertices for drawing polygons.
"""
from __future__ import annotations
from itertools import combinations
from math import atan2, gcd, sqrt
from functools import reduce


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def primitive(v):
    divisor = reduce(gcd, (abs(int(x)) for x in v), 0)
    return tuple(int(x)//divisor for x in v) if divisor else None


def cyclic_order(ids, points, normal):
    """Order facet vertices in its plane, solely for rendering faces correctly."""
    c = [sum(points[i][j] for i in ids)/len(ids) for j in range(3)]
    u = [points[ids[0]][j]-c[j] for j in range(3)]
    mag = sqrt(dot(u,u))
    if mag == 0:
        raise ValueError('Degenerate facet')
    u = [x/mag for x in u]
    w = cross(normal,u)
    wmag = sqrt(dot(w,w))
    if wmag == 0:
        raise ValueError('Degenerate facet normal')
    w = [x/wmag for x in w]
    return sorted(ids, key=lambda i: atan2(dot(subtract(points[i],c), w),
                                           dot(subtract(points[i],c), u)))


def exact_geometry(vertices):
    """Validate a full-dimensional, reflexive 3-polytope from exact vertices.

    Returns complete primal and dual graphs. No assumption that a coordinate
    normal form is comparable with a different polytope's normal form.
    """
    if not 4 <= len(vertices) <= 128 or any(len(v)!=3 or any(type(x) is not int for x in v) for v in vertices):
        raise ValueError('Expect 4..128 integral 3D vertices')
    if len(set(map(tuple,vertices)))!=len(vertices):
        raise ValueError('Duplicate vertices')
    vertices = [tuple(v) for v in vertices]
    faces = {}
    for i,j,k in combinations(range(len(vertices)),3):
        n = cross(subtract(vertices[j],vertices[i]),subtract(vertices[k],vertices[i]))
        if n == (0,0,0):
            continue
        reference = dot(n, vertices[i])
        differences = [dot(n,v)-reference for v in vertices]
        if min(differences) == 0 and max(differences) > 0:
            normal = primitive(n)
        elif max(differences) == 0 and min(differences) < 0:
            normal = primitive(tuple(-x for x in n))
        else:
            continue
        if min(dot(normal,v) for v in vertices) != -1:
            raise ValueError(f'Non-reflexive facet with primitive normal {normal}')
        faces[normal] = tuple(t for t,v in enumerate(vertices) if dot(normal,v)==-1)
    if len(faces)<4:
        raise ValueError('Polytope is not full dimensional')
    normals = sorted(faces)
    facet_ids = [list(faces[n]) for n in normals]
    if any(len(f)<3 for f in facet_ids):
        raise ValueError('Degenerate supporting facet')
    if any(not any(i in f for f in facet_ids) for i in range(len(vertices))):
        raise ValueError('Non-extreme input point')

    # For a 3-polytope, an edge is the intersection of at least two facets.
    edges = [[i,j] for i,j in combinations(range(len(vertices)),2)
             if sum(i in f and j in f for f in facet_ids)>=2]
    if len(vertices)-len(edges)+len(normals)!=2:
        raise ValueError('Euler characteristic invalid; input may contain non-vertices')
    facets = [cyclic_order(face, vertices, normal) for face,normal in zip(facet_ids,normals)]

    # Primal vertices become supporting hyperplanes of the dual.
    dual_facet_ids = [[j for j,n in enumerate(normals) if dot(n,v)==-1] for v in vertices]
    dual_facets = [cyclic_order(ids,normals,v) for ids,v in zip(dual_facet_ids,vertices)]
    dual_edges = [[i,j] for i,j in combinations(range(len(normals)),2)
                  if sum(i in f and j in f for f in dual_facet_ids)>=2]
    if len(normals)-len(dual_edges)+len(vertices)!=2:
        raise ValueError('Euler characteristic invalid for dual')
    if any(dot(n,v)<-1 for n in normals for v in vertices):
        raise ValueError('Polar incidence invalid')
    return {
      'vertices': [list(v) for v in vertices],
      'edges': edges,
      'facets': facets,
      'facet_normals': [list(n) for n in normals],
      'dual_vertices': [list(n) for n in normals],
      'dual_edges': dual_edges,
      'dual_facets': dual_facets,
      'vertex_count': len(vertices),
      'edge_count': len(edges),
      'facet_count': len(normals),
      'dual_vertex_count': len(normals),
      'dual_edge_count': len(dual_edges),
      'dual_facet_count': len(vertices),
    }
