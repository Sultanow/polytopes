#!/usr/bin/env python3
"""Exact, dependency-free verification of the bundled 2D atlas."""
import json
import unittest
from pathlib import Path
from generate_reflexive_2d import is_reflexive, area_twice, equivalent, inside, transforms, find_inclusion

FILE=Path(__file__).resolve().parents[1]/'src'/'data'/'reflexive-2d.json'
DATA=json.loads(FILE.read_text(encoding='utf-8'))
POLYS=DATA['polytopes']
BY_ID={p['id']:p for p in POLYS}
EDGES=DATA['cover_edges']

class TestReflexive2D(unittest.TestCase):
    def test_all_16_inequivalent_reflexive(self):
        self.assertEqual(len(POLYS),16)
        for poly in POLYS:
            vertices=tuple(tuple(v) for v in poly['vertices'])
            self.assertTrue(is_reflexive(vertices),poly['id'])
            self.assertEqual(area_twice(vertices),poly['area_twice'])
        for i,p in enumerate(POLYS):
            for q in POLYS[i+1:]:
                self.assertFalse(equivalent(tuple(map(tuple,p['vertices'])),tuple(map(tuple,q['vertices']))), (p['id'],q['id']))

    def test_inclusions_are_exact(self):
        for edge in EDGES:
            p,q=(BY_ID[edge[key]]['vertices'] for key in ('from','to'))
            a,b,c,d=edge['matrix']
            self.assertEqual(abs(a*d-b*c),1)
            self.assertTrue(all(inside((a*x+b*y,c*x+d*y),tuple(map(tuple,q))) for x,y in p),edge)
            self.assertLess(area_twice(tuple(map(tuple,p))),area_twice(tuple(map(tuple,q))))

    def test_no_intermediates_among_computed_pairs(self):
        mats=transforms()
        pairs=set()
        for p in POLYS:
            for q in POLYS:
                if p['area_twice']<q['area_twice'] and find_inclusion(p['vertices'],q['vertices'],mats):
                    pairs.add((p['id'],q['id']))
        for edge in EDGES:
            i,j=edge['from'],edge['to']
            self.assertFalse(any((i,k['id']) in pairs and (k['id'],j) in pairs for k in POLYS))

if __name__=='__main__':unittest.main()
