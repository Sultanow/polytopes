"""v0.14 source-shard manifest and bounds. All tests offline; no HF download."""
from __future__ import annotations
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from . import catalog4d_remote as r


class ShardCoverageV014Tests(unittest.TestCase):
    def empty_filters(self):
        return dict(facet_count=None, point_count=None,dual_point_count=None,
                    h11=None,h12=None,euler_characteristic=None)

    def test_manifest_is_exact_published_shard_list(self):
        manifest=r.remote_shards()
        expected=list(range(5,34))+[36]
        self.assertEqual(manifest['listed_shards'],expected)
        self.assertEqual(manifest['listed_shard_count'],30)
        self.assertEqual(manifest['missing_in_numeric_range'],[34,35])
        self.assertEqual(manifest['lookup_status'],'listed_in_repository_not_live_checked')
        self.assertFalse(manifest['all_shards_checked_for_current_filters'])
        self.assertEqual(manifest['total_classes_reference'],473800776)

    def test_36_vertex_shard_url_is_allowed_but_no_arbitrary_path(self):
        self.assertTrue(r.parquet_url(36).endswith('polytopes-4d-36-vertices.parquet'))
        self.assertTrue(r.parquet_url(16).endswith('polytopes-4d-16-vertices.parquet'))
        for bad in (4,34,35,37,-1,'36',False):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                r.parquet_url(bad)

    def test_new_large_shard_requires_a_filter(self):
        with self.assertRaises(HTTPException) as ctx:
            r.remote_count(vertex_count=16,**self.empty_filters())
        self.assertEqual(ctx.exception.status_code,422)
        with patch.object(r,'bounded_worker',return_value={'count':123,'count_exact_for_shard':True}) as worker:
            result=r.remote_count(vertex_count=16,**{**self.empty_filters(),'h11':101})
        self.assertEqual(result['count'],123)
        self.assertFalse(result['dataset_complete'])
        self.assertEqual(worker.call_args[0][0]['vertex_count'],16)

    def test_tail_shard_can_be_browsed_without_extra_filter(self):
        with patch.object(r,'bounded_worker',return_value={'items':[],'has_more':False,'next_cursor':None}) as worker:
            result=r.remote_search(vertex_count=36,limit=6,offset=0,after_row=None,**self.empty_filters())
        self.assertEqual(result['vertex_count'],36)
        self.assertEqual(worker.call_args[0][0]['mode'],'page')

    def test_invalid_shard_is_rejected_before_starting_worker(self):
        with patch.object(r,'bounded_worker') as worker:
            with self.assertRaises(HTTPException) as ctx:
                r.remote_row(vertex_count=34,shard_row=0,fp=None,rev=None)
            self.assertEqual(ctx.exception.status_code,422)
            worker.assert_not_called()

    def test_status_does_not_claim_global_coverage(self):
        status=r.remote_status()
        self.assertFalse(status['dataset_complete'])
        self.assertEqual(status['manifest_listed_shard_count'],30)
        self.assertEqual(status['max_parallel_queries'],1)
        self.assertEqual(status['timeout_seconds'],40)
        self.assertFalse(status['network_byte_cap'])


if __name__=='__main__':
    unittest.main()
