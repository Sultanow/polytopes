import unittest
from unittest.mock import patch
from fastapi import HTTPException
from . import catalog4d_live as live


class TestLive4D(unittest.TestCase):
    def test_allowlisted_predicates(self):
        self.assertEqual(live.predicates(h11=1, vertex_count=5), '"vertex_count"=5 AND "h11"=1')
        self.assertEqual(live.predicates(), '')
        with self.assertRaises(ValueError):
            live.predicates(h11='0 OR 1=1')
        with self.assertRaises(ValueError):
            live.predicates(vertex_count=1_000_001)

    def test_row_shape(self):
        sample={"row_idx":17,"row":{"vertices":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]],
                                         "vertex_count":5,"h11":101}}
        row=live.safe_row(sample)
        self.assertEqual(row['row_idx'],17)
        self.assertEqual(row['vertex_count'],5)
        self.assertEqual(len(row['vertices']),5)
        self.assertIsNone(row['h12'])
        self.assertIsNone(live.safe_row({'row_idx':0,'row':{'vertices':[[0,0,0]]}}))

    def test_remote_page_is_bounded(self):
        sample={"rows":[{"row_idx":1,"row":{"vertices":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]],"h11":101}}]}
        with patch.object(live, 'upstream_json', return_value=sample) as mock:
            result=live.search_page(limit=2, h11=101, vertex_count=5, offset=0)
            self.assertFalse(result['indexed_complete'])
            self.assertEqual(result['returned'],1)
            self.assertEqual(mock.call_args[0][0],'/filter')
            self.assertIn('"h11"=101',mock.call_args[0][1]['where'])

    def test_no_filter_uses_rows(self):
        with patch.object(live,'upstream_json',return_value={'rows':[]} ) as mock:
            result=live.search_page(limit=2,offset=0)
            self.assertEqual(result['returned'],0)
            self.assertEqual(mock.call_args[0][0],'/rows')

    def test_status_does_not_claim_completeness(self):
        with patch.object(live, 'upstream_json', return_value={'viewer': True, 'filter': True, 'search': False}):
            s=live.status()
        self.assertTrue(s['viewer'])
        self.assertFalse(s['indexed_complete'])
        self.assertFalse(s['search_supported_by_provider'])


if __name__=='__main__':
    unittest.main()
