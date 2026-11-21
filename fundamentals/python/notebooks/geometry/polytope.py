import numpy as np
from itertools import combinations
from cytools import Polytope as CP

def affine_rank(points, tol=1e-10):
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return 0
    base = pts[0]
    diffs = pts[1:] - base
    return int(np.linalg.matrix_rank(diffs, tol=tol))

def point_in_relative_interior(points, p):
    """
    Verification of the relative interior via the H-representation.
    """
    if points is None or len(points) == 0:
        return False
        
    # Convert points to integers and move them (p becomes the origin)
    pts_shifted = (np.array(points, dtype=int) - np.array(p, dtype=int)).tolist()
    poly = CP(pts_shifted)
    
    # Check the hyperplanes (inequalities)
    # The inequalities are in the form: a*x + b >= 0
    # A point x is in the strict interior if a*x + b > 0
    # For our shifted point x = [0,0,0,0,0], this reduces to: b > 0
    
    # Depending on the Cytools version, the method is called .inequalities() or .hyperplanes()
    hps = poly.inequalities()
            
    # hps is a matrix where the last column represents the offset 'b'
    offsets = hps[:, -1]
    
    # If all offsets are > 0, the origin lies strictly in the interior
    # (Note: b=0 would mean the point lies on the boundary)
    return np.all(offsets > 0)

def q_polytope_vertices_from_xpoints(x_points, tol=1e-10):
    """
    Polytope in q-space:
        q_i >= 0
        x^(j) · q = 1   for all j

    Vertices are basic solutions:
    choose rank(X) columns as basis, set others to 0, solve.
    """
    X = np.array(x_points, dtype=float)
    m, n = X.shape
    r = np.linalg.matrix_rank(X, tol=tol)

    if r != m:
        return []

    verts = []

    for basis in combinations(range(n), r):
        B = X[:, basis]
        if abs(np.linalg.det(B)) < tol:
            continue

        try:
            qB = np.linalg.solve(B, np.ones(m))
        except np.linalg.LinAlgError:
            continue

        q = np.zeros(n)
        q[list(basis)] = qB

        if np.all(q >= -tol) and np.allclose(X @ q, np.ones(m), atol=1e-9):
            q[q < 0] = 0.0
            verts.append(tuple(np.round(q, 14)))

    # deduplicate
    verts = sorted(set(verts))
    return [np.array(v, dtype=float) for v in verts]

def q_tilde_from_xpoints(x_points):
    verts = q_polytope_vertices_from_xpoints(x_points)
    if len(verts) != 1:
        return None, verts
    return verts[0], verts
