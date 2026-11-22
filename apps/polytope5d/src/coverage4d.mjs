/** Search-manifest bookkeeping. No network and no persistent server state. */
export const PUBLISHED_SHARDS = [...Array.from({length: 29}, (_, i) => i + 5), 36];
export const FILTER_REQUIRED_SHARDS = new Set(Array.from({length: 15}, (_, i) => i + 10));
const NUMERIC_FIELDS = ['facet_count', 'point_count', 'dual_point_count', 'h11', 'h12', 'euler_characteristic'];

export function signatureOf(filters, revision) {
  return [revision || 'main', ...NUMERIC_FIELDS.map(key => `${key}=${String(filters?.[key] ?? '')}`)].join('|');
}

export function anyNumericFilter(filters) {
  return NUMERIC_FIELDS.some(key => String(filters?.[key] ?? '') !== '');
}

export function summarizeCoverage(shards, entries = {}) {
  const exact = shards.filter(v => entries[v]?.state === 'exact' &&
    Number.isSafeInteger(entries[v]?.count) && entries[v]?.count >= 0);
  return {
    exactShardCount: exact.length,
    totalShards: shards.length,
    allExact: shards.length > 0 && exact.length === shards.length,
    exactCountSubtotal: exact.reduce((sum, v) => sum + entries[v].count, 0),
  };
}
