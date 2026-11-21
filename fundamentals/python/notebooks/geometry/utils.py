from math import gcd
from functools import reduce
from fractions import Fraction

def rationalize_q(q, max_den=200000):
    return tuple(Fraction(float(v)).limit_denominator(max_den) for v in q)

def gcd_many(xs): 
    """GCD of a list of numbers."""
    return gcd(*xs)

def lcm(a, b): 
    """Least common multiple of two numbers."""
    return abs(a * b) // gcd(a, b)

def lcm_many(nums): 
    """Least common multiple of a list."""
    nums = [abs(int(n)) for n in nums if int(n) != 0]
    if not nums:
        return 1
    return reduce(lcm, nums, 1)

def is_primitive(v):
    """Primitive normals?"""
    vals = [abs(int(x)) for x in v if int(x) != 0]
    if not vals:
        return False
    return reduce(gcd, vals) == 1

def dominates_lex_desc(x, y): 
    """Lexicographical comparison in descending order."""
    return tuple(x) > tuple(y)

def canonicalize_rows(matrix_tuple):
    r1, r2 = (tuple(int(x) for x in row) for row in matrix_tuple)
    return (r1, r2) if r1 <= r2 else (r2, r1)

def build_55_overlap3_from_5weights(ws1, ws2):
    return ((ws1[0], ws1[1], ws1[2], ws1[3], ws1[4], 0, 0),
            (ws2[0], ws2[1], ws2[2], 0, 0, ws2[3], ws2[4]))