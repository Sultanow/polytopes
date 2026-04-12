# geometry/__init__.py

import cytools
cytools.config.enable_experimental_features()

# 1. Basis-Mathematik & Hilfsmittel
from .utils import (
    gcd_many,
    lcm_many,
    rationalize_q,
    dominates_lex_desc,
    canonicalize_rows,
    build_55_overlap3_from_5weights
)

# 2. Gitterpunkte & Backtracking
from .lattice import (
    enumerate_nonnegative_solutions,
    points_from_weights,
    lattice_points_from_matrix,
    integer_kernel_basis,
    lattice_coordinates_in_kernel_basis,
    reflexive_in_cws_lattice
)

# 3. Geometrische Analyse
from .polytope import (
    affine_rank,
    point_in_relative_interior,
    q_polytope_vertices_from_xpoints,
    q_tilde_from_xpoints
)

# 4. CWS-Spezifische Logik
from .cws_search import (
    ReducedWSRecord,
    reduced_q_to_integer_weights,
    lattice_points_for_reduced_q,
    has_ip_reduced_q,
    lift_reduced_5_weights_to_sextuple,
    generate_reduced_5_ws_via_x,
    analyze_cws_candidate
)