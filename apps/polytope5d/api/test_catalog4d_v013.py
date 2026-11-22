"""v0.13 cursor and deep-link regression tests; no network required."""
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from fastapi import HTTPException
from . import catalog4d_remote as remote


class ResearchExplorerV013Tests(unittest.TestCase):
    def sample(self):
        return {
            "shard_row": 42, "vertex_count": 5, "facet_count": 5,
            "point_count": 6, "dual_point_count": 26, "h11": 21,
            "h12": 1, "euler_characteristic": 40,
            "vertices": [[1,0,0,0],[0,1,0,0],[0,0,1,0],
                         [1,2,3,5],[-2,-3,-4,-5]],
        }

    def test_fingerprint_is_deterministic_and_sensitive(self):
        x = self.sample()
        fp = remote.row_fingerprint(x)
        self.assertEqual(len(fp), 64)
        self.assertEqual(fp, remote.row_fingerprint(dict(reversed(list(x.items())))))
        changed = dict(x)
        changed['h11'] = 22
        self.assertNotEqual(fp, remote.row_fingerprint(changed))
        self.assertEqual(fp, remote.add_fingerprint(x)['fingerprint'])

    def test_valid_fingerprint_link_is_accepted(self):
        item = remote.add_fingerprint(self.sample())
        with patch.object(remote, 'bounded_worker', return_value={"item":item}):
            result = remote.remote_row(vertex_count=5, shard_row=42,
                                       fp=item['fingerprint'], rev=None)
        self.assertEqual(result['item']['shard_row'],42)
        self.assertEqual(result['item']['fingerprint'],item['fingerprint'])

    def test_changed_link_fails_closed(self):
        item = remote.add_fingerprint(self.sample())
        with patch.object(remote, 'bounded_worker', return_value={"item":item}):
            with self.assertRaises(HTTPException) as ctx:
                remote.remote_row(vertex_count=5, shard_row=42,
                                  fp='a'*64, rev=None)
        self.assertEqual(ctx.exception.status_code,409)

    def test_cross_revision_link_rejected_before_query(self):
        wrong = '0'*40 if remote.SOURCE_REVISION != '0'*40 else '1'*40
        with patch.object(remote, 'bounded_worker') as worker:
            with self.assertRaises(HTTPException) as ctx:
                remote.remote_row(vertex_count=5, shard_row=42, fp=None, rev=wrong)
            worker.assert_not_called()
        self.assertEqual(ctx.exception.status_code,409)

    def test_cursor_passed_to_worker(self):
        with patch.object(remote, 'bounded_worker', return_value={'items':[],'has_more':False,'next_cursor':None}) as worker:
            response = remote.remote_search(vertex_count=8,offset=0,after_row=10005,limit=6,
                facet_count=None, point_count=None, dual_point_count=None,
                h11=101,h12=None,euler_characteristic=None)
        self.assertEqual(worker.call_args[0][0]['after_row'],10005)
        self.assertFalse(response['dataset_complete'])

    def test_offset_and_cursor_are_mutually_exclusive(self):
        with self.assertRaises(HTTPException) as ctx:
            remote.remote_search(vertex_count=8, offset=1,after_row=123,limit=6,
                facet_count=None, point_count=None, dual_point_count=None,
                h11=101,h12=None,euler_characteristic=None)
        self.assertEqual(ctx.exception.status_code,422)

    def test_sql_cursor_uses_file_row_number_not_offset(self):
        connection = MagicMock()
        connection.execute.return_value = connection
        connection.fetchall.return_value = []
        with patch.dict('sys.modules', {'duckdb':SimpleNamespace(connect=lambda _:connection)}):
            with patch.object(remote, 'parquet_url', return_value='/tmp/sample.parquet'):
                result = remote._duckdb_worker({'mode':'page','vertex_count':5,
                    'filters':{'h11':101},'after_row':10005,'limit':6})
        self.assertEqual(result['items'],[])
        sql_args = [c.args for c in connection.execute.call_args_list]
        matching = [args for args in sql_args if args and 'file_row_number > ?' in args[0]]
        self.assertEqual(len(matching),1)
        self.assertIn(10005, matching[0][1])
        self.assertNotIn('OFFSET', matching[0][0])

    def test_status_does_not_claim_full_dataset_or_pinned_main(self):
        status = remote.remote_status()
        self.assertFalse(status['dataset_complete'])
        self.assertEqual(status['source_revision'],remote.SOURCE_REVISION)
        self.assertEqual(status['source_revision_pinned'],remote.REVISION_PINNED)
        self.assertFalse(status['network_byte_cap'])


if __name__ == '__main__':
    unittest.main()
