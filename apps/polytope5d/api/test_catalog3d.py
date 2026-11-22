import tempfile
import unittest
from pathlib import Path
from .catalog3d_geometry import exact_geometry
from .catalog3d_import import read_palp_matrices, import_text
from .catalog3d_store import ensure_schema
from .db import connect

CUBE=[[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)]
OCTAHEDRON=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
SIMPLEX=[[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]]


def to_palp(vertices):
    return f'3 {len(vertices)} Vertices\n'+'\n'.join(' '.join(str(v[i]) for v in vertices) for i in range(3))+'\n'

class ExactThreeDTests(unittest.TestCase):
    def test_examples_and_polar_duality(self):
        for vertices,edges,facets,dual_vertices in [(CUBE,12,6,6),(OCTAHEDRON,12,8,8),(SIMPLEX,6,4,4)]:
            with self.subTest(vertices=len(vertices)):
                g=exact_geometry(vertices)
                self.assertEqual((g['edge_count'],g['facet_count'],g['dual_vertex_count']),
                                 (edges,facets,dual_vertices))
                self.assertEqual(g['vertex_count']-g['edge_count']+g['facet_count'],2)
                d=exact_geometry(g['dual_vertices'])
                self.assertEqual(len(d['facet_normals']),len(vertices))
                self.assertEqual(sorted(map(tuple,d['facet_normals'])),sorted(map(tuple,vertices)))
                self.assertTrue(all(len(face)>=3 for face in g['facets']))
    def test_bad_non_reflexive(self):
        with self.assertRaises(ValueError):
            exact_geometry([[2,0,0],[0,2,0],[0,0,2],[-2,-2,-2]])
    def test_matrix_format(self):
        text=to_palp(CUBE)+to_palp(OCTAHEDRON)
        self.assertEqual(list(read_palp_matrices(text)),[CUBE,OCTAHEDRON])
    def test_transactional_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'test.db'
            with self.assertRaises(ValueError):
                import_text(to_palp(CUBE),db,expected_count=4319)
            ensure_schema(db)
            with connect(db) as con:
                self.assertEqual(con.execute('SELECT COUNT(*) FROM catalogue3d').fetchone()[0],0)
            result=import_text(to_palp(CUBE)+to_palp(OCTAHEDRON),db,expected_count=2)
            self.assertEqual(result['count'],2)
            with connect(db) as con:
                self.assertEqual(con.execute('SELECT COUNT(*) FROM catalogue3d').fetchone()[0],2)
                self.assertEqual(con.execute('SELECT COUNT(*) FROM catalogue3d_provenance').fetchone()[0],4)

if __name__=='__main__':
    unittest.main()
