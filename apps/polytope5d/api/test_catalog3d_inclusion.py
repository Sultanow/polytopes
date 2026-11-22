"""Certificate tests independent of any externally sourced catalogue mapping."""
import unittest
from .catalog3d_inclusion import certificate_valid, signed_permutation_matrices, det3

CUBE=[[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)]
OCTA=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
CUBE_NORMALS=OCTA
OCTA_NORMALS=CUBE
I=[[1,0,0],[0,1,0],[0,0,1]]

class InclusionCertificateTests(unittest.TestCase):
    def test_signed_permutation_group(self):
        matrices=list(signed_permutation_matrices())
        self.assertEqual(len(matrices),48)
        self.assertEqual(len({tuple(sum(m,[])) for m in matrices}),48)
        self.assertTrue(all(abs(det3(m))==1 for m in matrices))

    def test_octa_inside_cube(self):
        self.assertTrue(certificate_valid(OCTA,CUBE,CUBE_NORMALS,I))
        self.assertFalse(certificate_valid(CUBE,OCTA,OCTA_NORMALS,I))

    def test_equality_is_not_strict(self):
        self.assertFalse(certificate_valid(CUBE,CUBE,CUBE_NORMALS,I))
        self.assertTrue(certificate_valid(CUBE,CUBE,CUBE_NORMALS,I,strict=False))

    def test_non_unimodular_rejected(self):
        self.assertFalse(certificate_valid(OCTA,CUBE,CUBE_NORMALS,[[2,0,0],[0,1,0],[0,0,1]]))

    def test_unimodular_rotation_works(self):
        u=[[0,-1,0],[1,0,0],[0,0,1]]
        self.assertTrue(certificate_valid(OCTA,CUBE,CUBE_NORMALS,u))

if __name__=='__main__':
    unittest.main()
