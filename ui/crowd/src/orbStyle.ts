/** Mirrors M29 rows in params/registry.csv. tests/test_m29_orb_aggregate.py locks the values. */

export const ORB_BASE_RADIUS = 2.1;
export const ORB_UNINSURED_EXTRA = 1.9;
export const ORB_RADIUS_CAP = 5.2;
export const ORBS_PER_TICK = 5;
export const MAX_DRAWN_ORBS = 2048;

export function blockSize(totalPeople: number, orbCount: number): number {
  if (totalPeople < 1 || orbCount < 1) {
    return 1;
  }
  return Math.max(1, Math.round(totalPeople / orbCount));
}

/** Pixel radius. Balance is clamped to 0..1 so a bad payload cannot explode the dot. */
export function orbRadius(uninsured: boolean, balanceNorm: number): number {
  const bal = Math.min(1, Math.max(0, balanceNorm));
  const radius = ORB_BASE_RADIUS + bal * (ORB_BASE_RADIUS * 0.65) + (uninsured ? ORB_UNINSURED_EXTRA : 0);
  return Math.min(ORB_RADIUS_CAP, radius);
}

const STATE_RGB: [number, number, number][] = [
  [62, 196, 112],
  [168, 176, 186],
  [232, 96, 104],
];

export function stateRgb(state: number): [number, number, number] {
  return STATE_RGB[state] ?? STATE_RGB[0];
}
