"""Local Parquet smoke test; skipped when DuckDB is absent. Never calls the internet."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from . import catalog4d_remote as remote

try:
    import duckdb
except ImportError:
    duckdb = None


@unittest.skipIf(duckdb is None, "DuckDB not installed in this Python runtime")
class LocalParquetTest(unittest.TestCase):
    def test_real_duckdb_file_row_numbers_count_and_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'polytope-sample.parquet')
            con = duckdb.connect(':memory:')
            try:
                con.execute('''CREATE TABLE p AS SELECT
                    [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]] AS vertices,
                    5 AS vertex_count, 5 AS facet_count, 6 AS point_count,
                    6 AS dual_point_count, 101 AS h11, 1 AS h12,
                    200 AS euler_characteristic''')
                con.execute('INSERT INTO p SELECT vertices, vertex_count, facet_count, point_count, dual_point_count, 102 AS h11, h12, euler_characteristic FROM p LIMIT 1')
                safe_path=path.replace("'","''")
                con.execute("COPY p TO '" + safe_path + "' (FORMAT PARQUET)")
            finally:
                con.close()
            with patch.object(remote, 'parquet_url', return_value=path):
                count = remote._duckdb_worker({'mode':'count','vertex_count':5,'filters':{'h11':101}})
                self.assertEqual(count['count'],1)
                page = remote._duckdb_worker({'mode':'page','vertex_count':5,
                    'filters':{'h11':101},'offset':0,'limit':6})
                self.assertEqual(len(page['items']),1)
                row = page['items'][0]
                self.assertEqual(row['shard_row'],0)
                self.assertEqual(len(row['vertices']),5)
                opened = remote._duckdb_worker({'mode':'row','vertex_count':5,'shard_row':0})
                self.assertEqual(opened['item']['h11'],101)
