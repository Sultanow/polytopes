"""Run: python -m unittest api.test_api api.test_duality -v"""
import tempfile
import unittest
from pathlib import Path

from .db import connect, demo_records, init_db
from .duality import dot, enrich_detail, exact_dual


class ExactPolarTests(unittest.TestCase):
    def test_all_69_demo_duals(self):
        demos = list(demo_records())
        self.assertEqual(len(demos), 69)
        for record in demos:
            with self.subTest(record=record["id"]):
                data = exact_dual(record)
                self.assertTrue(data["dual_available"])
                self.assertEqual(data["dual_vertex_count"], record["facet_count"])
                self.assertEqual(data["dual_facet_count"], record["vertex_count"])
                self.assertEqual(len(data["facets"]), record["facet_count"])
                for normal, facet in zip(data["dual_vertices_5d"], data["facets"]):
                    evaluations = [dot(normal, v) for v in record["vertices_5d"]]
                    self.assertTrue(all(value >= -1 for value in evaluations))
                    self.assertEqual(facet, [i for i, val in enumerate(evaluations) if val == -1])
                    self.assertGreaterEqual(len(facet), 5)
                for primal, facet in zip(record["vertices_5d"], data["dual_facets"]):
                    self.assertEqual(facet, [i for i,u in enumerate(data["dual_vertices_5d"])
                                             if dot(primal, u) == -1])
                    self.assertGreaterEqual(len(facet), 5)

    def test_special_dual_graphs(self):
        by_id = {r["id"]: r for r in demo_records()}
        cube = exact_dual(by_id["demo-hypercube-5d"])
        cross = exact_dual(by_id["demo-crosspolytope-5d"])
        self.assertEqual((len(cube["dual_vertices_5d"]), len(cube["dual_edges"])), (10, 40))
        self.assertEqual((len(cross["dual_vertices_5d"]), len(cross["dual_edges"])), (32, 80))
        self.assertEqual(cube["dual_vertices_5d"], by_id["demo-crosspolytope-5d"]["vertices_5d"])
        self.assertEqual(cross["dual_vertices_5d"], by_id["demo-hypercube-5d"]["vertices_5d"])

    def test_never_claim_dual_for_import(self):
        imported = {"id": "catalogue-1", "source": "catalogue_excerpt",
                    "vertices_5d": [[0, 0, 0, 0, 0]], "facet_normals": [[1]*5]}
        self.assertIsNone(exact_dual(imported))
        self.assertFalse(enrich_detail(imported)["dual_available"])

    def test_seeded_persistent_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "polytopes.sqlite3"
            init_db(db)
            with connect(db) as con:
                row = con.execute("SELECT * FROM polytopes WHERE id='demo-hypercube-5d'").fetchone()
                self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
