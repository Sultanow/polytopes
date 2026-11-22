"""Offline tests for v0.12. Real HTTPFS/Parquet runs are verified on the target host."""
from __future__ import annotations
import json
from unittest import TestCase
from unittest.mock import patch
from fastapi import HTTPException

from . import catalog4d_remote as r


class Remote4DTests(TestCase):
    def test_shard_url_cannot_be_user_supplied(self):
        self.assertTrue(r.parquet_url(15).endswith("polytopes-4d-15-vertices.parquet"))
        for bad in (4, 16, "15", "../x", True):
            with self.assertRaises(ValueError):
                r.parquet_url(bad)

    def test_filters_are_whitelisted_and_parameterized(self):
        sql, params = r.filters_for_sql(12, {"h11":101,"facet_count":20,"foobar":"inject"})
        self.assertEqual(params,[12,20,101])
        self.assertIn("h11 = ?",sql)
        self.assertNotIn("foobar",sql)

    def test_invalid_filters(self):
        for val in ("100", True, 2_000_000, 1.5):
            with self.assertRaises(ValueError):
                r.filters_for_sql(8,{"h11":val})

    def test_status_accurately_discloses_limits(self):
        s=r.remote_status()
        self.assertEqual(s['max_parallel_queries'],1)
        self.assertFalse(s['dataset_complete'])
        self.assertFalse(s['network_byte_cap'])
        self.assertEqual(s['timeout_seconds'],40)

    def test_count_request_is_distinct_from_all_catalogue(self):
        with patch.object(r,'bounded_worker',return_value={"count":9828,"count_exact_for_shard":True}) as mock:
            ans=r.remote_count(vertex_count=15, h11=101, facet_count=None,
                               point_count=None, dual_point_count=None,h12=None,euler_characteristic=None)
        self.assertEqual(ans['count'],9828)
        self.assertFalse(ans['dataset_complete'])
        self.assertEqual(mock.call_args[0][0]['vertex_count'],15)
        self.assertEqual(mock.call_args[0][0]['filters']['h11'],101)

    def test_unfiltered_large_scan_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            r.remote_search(vertex_count=15,offset=0,limit=6,facet_count=None,
                point_count=None,dual_point_count=None,h11=None,h12=None,euler_characteristic=None)
        self.assertEqual(error.exception.status_code,422)

    def test_page_is_shard_local(self):
        with patch.object(r,'bounded_worker',return_value={"items":[{"shard_row":48,"vertices":[[0]*4]*5}],"has_more":True}) as mock:
            ans=r.remote_search(vertex_count=8,offset=0,limit=6,facet_count=None,
                point_count=None,dual_point_count=None,h11=101,h12=None,euler_characteristic=None)
        self.assertEqual(ans['items'][0]['shard_row'],48)
        self.assertEqual(mock.call_args[0][0]['mode'],'page')
        self.assertFalse(ans['dataset_complete'])

    def test_worker_timeout_kills_subprocess(self):
        import subprocess
        with patch.object(r.subprocess,'run',side_effect=subprocess.TimeoutExpired('cmd',40)):
            with self.assertRaises(HTTPException) as e:
                r.bounded_worker({'mode':'count'})
        self.assertEqual(e.exception.status_code,504)

    def test_worker_reports_busy(self):
        with patch.object(r.fcntl,'flock',side_effect=BlockingIOError()):
            with self.assertRaises(HTTPException) as e:
                r.bounded_worker({'mode':'count'})
        self.assertEqual(e.exception.status_code,429)

    def test_bounded_worker_parses_short_json(self):
        from types import SimpleNamespace
        child=SimpleNamespace(returncode=0,stdout=json.dumps({'count':21}),stderr='')
        with patch.object(r.subprocess,'run',return_value=child):
            self.assertEqual(r.bounded_worker({'mode':'count'})['count'],21)

    def test_source_row_does_not_claim_global_id(self):
        with patch.object(r,'bounded_worker',return_value={'item':{'shard_row':123}}):
            item=r.remote_row(vertex_count=9,shard_row=123)
        self.assertEqual(item['identity'],'(vertex_count, shard_row)')
