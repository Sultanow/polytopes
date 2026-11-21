import numpy as np
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Optional, List, Tuple

from .utils import (
    gcd_many, 
    lcm_many, 
    rationalize_q, 
    dominates_lex_desc, 
)
from .lattice import (
    points_from_weights, 
    enumerate_nonnegative_solutions,
    lattice_points_from_matrix,
    reflexive_in_cws_lattice
)
from .polytope import (
    point_in_relative_interior, 
    q_tilde_from_xpoints,
    q_polytope_vertices_from_xpoints,
    affine_rank
)

from cytools import Polytope as CP

@dataclass(frozen=True)
class ReducedWSRecord:
    x_points: tuple[tuple[int, ...], ...]
    q_tilde: tuple[float, ...]
    int_weights_5: tuple[int, ...]
    sextuple_weights: tuple[int, ...]

def reduced_q_to_integer_weights(q_frac):
    """
    q has sum 1/2 in reduced 5-weight form.
    Convert to primitive integer weights (w1,...,w5) with sum = D/2.
    """
    den = lcm_many([f.denominator for f in q_frac])
    ints = [int(f * den) for f in q_frac]
    g = gcd_many(ints)
    ints = [x // g for x in ints]
    return tuple(ints)

def lattice_points_for_reduced_q(q_frac):
    """
    Reduced case: q has length 5, sum(q)=1/2.
    We enumerate x >= 0 integer with x·q = 1.
    """
    w = reduced_q_to_integer_weights(q_frac)
    pts = points_from_weights([w], degrees=[sum(w)*2])
    return [np.array(p) for p in pts]

def has_ip_reduced_q(q_frac):
    pts = lattice_points_for_reduced_q(q_frac)
    p = np.array([2, 2, 2, 2, 2], dtype=float)
    return point_in_relative_interior(pts, p)
    
def enumerate_x_candidates(q_tilde, tol=1e-12, max_candidates=200):
    """
    Finds canonical representatives x in Z_{>=0}^n that satisfy x · q_tilde < 1.

    Symmetry Breaking:
        Enforces x1 >= x2 >= ... >= xn to eliminate permutations.

    Additional Filters (Hard-coded):
        - Excludes the zero vector.
        - Excludes vectors with sum <= 2.
        - Excludes vectors where all entries are <= 2 (filters out very small weights).

    Sorting:
        1. Ascending by the value of x · q_tilde (closer to the boundary first).
        2. Descending lexicographical order (prefers "top-heavy" vectors).

    Parameters
    ----------
    q_tilde : array_like
        The current reference point in q-space.
    tol : float
        Numerical tolerance for the budget constraint.
    max_candidates : int, optional
        Maximum number of top candidates to return.

    Returns
    -------
    list[tuple]
        A sorted list of integer lattice points.
    """
    if q_tilde is None:
        return []

    q = np.array(q_tilde, dtype=float)
    if q.ndim != 1:
        return []

    q[q < tol] = 0.0
    n = len(q)

    positive_q = q[q > tol]
    if len(positive_q) == 0:
        return []

    qmin = np.min(positive_q)
    max_total = int(np.floor((1.0 - 1e-12) / qmin))

    out = []
    x = [0] * n

    def rec(i, remaining_budget, prev_val):
        if i == n:
            val = float(np.dot(x, q))
            if val < 1.0 - 1e-12:
                out.append(tuple(x))
            return

        if q[i] > tol:
            ub = int(np.floor((remaining_budget - 1e-12) / q[i]))
        else:
            ub = max_total

        ub = min(ub, prev_val)

        for v in range(ub, -1, -1):
            x[i] = v
            new_budget = remaining_budget - v * q[i]
            if new_budget < -1e-12:
                continue
            rec(i + 1, new_budget, v)

        x[i] = 0

    rec(0, 1.0 - 1e-12, max_total)

    filtered = []
    for cand in out:
        if cand == (0, 0, 0, 0, 0):
            continue
        if sum(cand) <= 2:
            continue
        if all(v <= 2 for v in cand):
            continue
        filtered.append(cand)

    filtered = sorted(
        filtered,
        key=lambda x: (float(np.dot(x, q)), tuple(-v for v in x))
    )

    if max_candidates is not None:
        filtered = filtered[:max_candidates]

    return filtered

def lift_reduced_5_weights_to_sextuple(w5):
    D2 = sum(w5)
    full = (D2,) + tuple(w5)
    g = gcd_many(full)
    return tuple(x // g for x in full)

def q_search_point_from_xpoints(x_points, tol=1e-10):
    """
    Determine the current q-polytope from x_points.

    Returns
    -------
    q_search : np.ndarray or None
        A representative point inside the current q-polytope.
        If the polytope is a single point, this is that unique point.
        Otherwise we use the barycenter of vertices as a search point.
    verts : list[np.ndarray]
        Vertices of the current q-polytope.
    is_unique : bool
        True iff the q-polytope has collapsed to a single point.
    """
    verts = q_polytope_vertices_from_xpoints(x_points, tol=tol)
    if not verts:
        return None, [], False

    if len(verts) == 1:
        return np.array(verts[0], dtype=float), verts, True

    q_search = np.mean(np.array(verts, dtype=float), axis=0)
    return q_search, verts, False

def generate_reduced_5_ws_via_x(max_depth=5, verbose=True, max_results=None):
    start = ((2, 2, 2, 2, 2),)
    results = []
    seen_weights = set()
    visited_xconfigs = set()

    def recurse(x_points):
        if max_results is not None and len(results) >= max_results:
            return

        x_key = tuple(sorted(x_points))
        if x_key in visited_xconfigs:
            return
        visited_xconfigs.add(x_key)

        q_search, verts, is_unique = q_search_point_from_xpoints(x_points)
        if q_search is None:
            return

        # Nur wenn das q-Polytope wirklich ein Punkt ist,
        # darf daraus ein echtes reduziertes Gewichtssystem werden.
        if is_unique:
            q_frac = rationalize_q(q_search)
            w5 = reduced_q_to_integer_weights(q_frac)

            if w5 not in seen_weights and has_ip_reduced_q(q_frac):
                seen_weights.add(w5)
                results.append(
                    ReducedWSRecord(
                        x_points=x_points,
                        q_tilde=tuple(float(v) for v in q_search),
                        int_weights_5=w5,
                        sextuple_weights=lift_reduced_5_weights_to_sextuple(w5),
                    )
                )
                if verbose:
                    print(f"[{len(results)}] Treffer gefunden: {w5}")

        if len(x_points) >= max_depth:
            return

        cands = enumerate_x_candidates(q_search)
        old_rank = np.linalg.matrix_rank(np.array(x_points, dtype=float))

        for x in cands:
            if x in x_points:
                continue

            Xnew = np.array(x_points + (x,), dtype=float)
            if np.linalg.matrix_rank(Xnew) > old_rank:
                recurse(x_points + (x,))

    recurse(start)
    return results

def analyze_cws_candidate(matrix_tuple, target=None):
    if target is None:
        target = np.ones(len(matrix_tuple[0]), dtype=int)

    W = np.array(matrix_tuple, dtype=int)
    degrees = W.sum(axis=1)

    if not np.array_equal(W @ target, degrees):
        return None

    pts = lattice_points_from_matrix(matrix_tuple)
    if len(pts) == 0:
        return None

    pts_arr = np.array(pts, dtype=int)
    if not point_in_relative_interior(pts_arr, target):
        return None

    is_refl, poly5d, pts5d = reflexive_in_cws_lattice(matrix_tuple)

    poly_dim = affine_rank(pts5d) if pts5d is not None and len(pts5d) > 0 else None    
    if poly_dim != 5:
        return None
    
    return {
        "n_lattice_points": len(pts),
        "dim": poly_dim,
        "reflexive": is_refl,
        "pts5d_count": len(pts5d) if pts5d is not None else None,
    }
