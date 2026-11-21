import json
import tempfile
import unittest
from pathlib import Path
from .db import connect, demo_records, deserialize, init_db, normalize_record, simplex
from .import_jsonl import import_file

class MathsAndIndexTests(unittest.TestCase):
  def test_simplex_facets_exact(self):
    for q in ((1,1,1,1,1),(1,1,1,2,2),(1,1,2,3,4)):
      rec=simplex(q)
      vertices=rec["vertices_5d"]
      for normal in rec["facet_normals"]:
        products=[sum(a*b for a,b in zip(normal, v)) for v in vertices]
        self.assertEqual(sum(v==-1 for v in products),5)
        self.assertTrue(all(v>=-1 for v in products))
  def test_two_dual_reference_shapes(self):
    rec={r["id"]:r for r in demo_records()}
    self.assertEqual(len(rec),69)
    self.assertEqual((rec['demo-hypercube-5d']['vertex_count'],len(rec['demo-hypercube-5d']['edges'])),(32,80))
    self.assertEqual((rec['demo-crosspolytope-5d']['vertex_count'],len(rec['demo-crosspolytope-5d']['edges'])),(10,40))
  def test_import_and_search_index(self):
    with tempfile.TemporaryDirectory() as d:
      db=Path(d)/'test.sqlite'; src=Path(d)/'test.jsonl'
      init_db(db)
      data={"id":"catalogue-example-1","source":"user_subset","dimension":5,"reflexive":True,
            "weights":[1,1,1,1,1,1], "vertex_count":6,"facet_count":6}
      src.write_text(json.dumps(data)+'\n')
      self.assertEqual(import_file(src,db),1)
      with connect(db) as con:
        self.assertEqual(con.execute("SELECT COUNT(*) FROM polytopes").fetchone()[0],70)
        fetched=deserialize(con.execute("SELECT * FROM polytopes WHERE id=?",('catalogue-example-1',)).fetchone())
      self.assertEqual(fetched['weights'],[1]*6)
      self.assertEqual(fetched['source'],'user_subset')
  def test_invalid_nonreflexive_and_fabricated_geometry(self):
    with self.assertRaises(ValueError):normalize_record({'id':'x','source':'user_subset','reflexive':'yes'})
    with self.assertRaises(ValueError):normalize_record({'id':'x','source':'user_subset','reflexive':True,
                  'vertices_5d':[[1,2,3]],'vertex_count':1})

if __name__=='__main__':unittest.main()
