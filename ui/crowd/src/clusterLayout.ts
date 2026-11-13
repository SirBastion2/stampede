/** Mirrors `ui/crowd/layout.py` community indexing (deterministic, no extra frame fields). */

export const DEFAULT_COMMUNITY_COUNT = 8;

export function communityIds(pointCount: number, communityCount: number): number[] {
  const communities = Math.min(communityCount, pointCount);
  const per = Math.floor(pointCount / communities);
  const remainder = pointCount % communities;
  const ids: number[] = [];
  for (let c = 0; c < communities; c += 1) {
    const count = per + (c < remainder ? 1 : 0);
    for (let i = 0; i < count; i += 1) ids.push(c);
  }
  return ids;
}

export type ClusterCenter = { id: number; x: number; y: number };

export function clusterCenters(
  positions: number[],
  ids: number[],
  communityCount: number,
): ClusterCenter[] {
  const sums = Array.from({ length: communityCount }, () => ({ x: 0, y: 0, n: 0 }));
  for (let i = 0; i < ids.length; i += 1) {
    const c = ids[i];
    sums[c].x += positions[i * 2];
    sums[c].y += positions[i * 2 + 1];
    sums[c].n += 1;
  }
  return sums.map((row, id) => ({
    id,
    x: row.n ? row.x / row.n : 0,
    y: row.n ? row.y / row.n : 0,
  }));
}

export type ClusterLink = { from: number; to: number; longRange: boolean };

/** Adjacent communities on the layout ring plus seeded long-range jumps. */
export function clusterLinks(seed: number, communityCount: number): ClusterLink[] {
  const links: ClusterLink[] = [];
  for (let i = 0; i < communityCount; i += 1) {
    links.push({ from: i, to: (i + 1) % communityCount, longRange: false });
  }
  const jumpA = (seed * 17 + 3) % communityCount;
  const jumpB = (seed * 13 + 5) % communityCount;
  if (jumpA !== jumpB) {
    links.push({ from: jumpA, to: jumpB, longRange: true });
  }
  const jumpC = (seed * 11 + 7) % communityCount;
  const jumpD = (jumpC + Math.floor(communityCount / 2)) % communityCount;
  if (jumpC !== jumpD) {
    links.push({ from: jumpC, to: jumpD, longRange: true });
  }
  return links;
}

export type ClusterStats = {
  id: number;
  calm: number;
  wavering: number;
  withdrew: number;
  total: number;
  withdrewShare: number;
};

export function clusterStats(
  state: number[],
  ids: number[],
  communityCount: number,
): ClusterStats[] {
  const rows = Array.from({ length: communityCount }, (_, id) => ({
    id,
    calm: 0,
    wavering: 0,
    withdrew: 0,
    total: 0,
    withdrewShare: 0,
  }));
  for (let i = 0; i < state.length; i += 1) {
    const c = ids[i] ?? 0;
    const s = state[i] ?? 0;
    rows[c].total += 1;
    if (s === 2) rows[c].withdrew += 1;
    else if (s === 1) rows[c].wavering += 1;
    else rows[c].calm += 1;
  }
  for (const row of rows) {
    row.withdrewShare = row.total ? row.withdrew / row.total : 0;
  }
  return rows;
}

/** Seeded origin cluster for spread narrative (matches layout ring). */
export function outbreakOriginCluster(seed: number, communityCount: number): number {
  return ((seed % communityCount) + communityCount) % communityCount;
}
