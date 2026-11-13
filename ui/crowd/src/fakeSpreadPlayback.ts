import {
  communityIds,
  DEFAULT_COMMUNITY_COUNT,
  outbreakOriginCluster,
} from "./clusterLayout";
import { blockSize } from "./orbStyle";
import type { CrowdFrame } from "./types";

const STATE_CALM = 0;
const STATE_WAVERING = 1;
const STATE_WITHDREW = 2;

function cycleSeconds(): number {
  return 105;
}

/** Deterministic cluster wave from origin; loops for fake mode. */
export function fakeSpreadFrame(
  base: CrowdFrame,
  elapsedSec: number,
  rumorText: string,
  effectScale = 1,
): CrowdFrame {
  if (effectScale <= 0) {
    return {
      ...base,
      panels: {
        ...base.panels,
        rumor: rumorText || base.panels.rumor,
      },
    };
  }
  const communityCount = DEFAULT_COMMUNITY_COUNT;
  const ids = communityIds(base.point_count, communityCount);
  const origin = outbreakOriginCluster(base.seed, communityCount);
  const cycle = cycleSeconds();
  const phase = (elapsedSec % cycle) / cycle;
  const progress = Math.min(1, phase * 1.05 * Math.max(0, Math.min(1, effectScale)));

  const clusterWithdrawShare = Array.from({ length: communityCount }, () => 0);
  for (let c = 0; c < communityCount; c += 1) {
    const ringDist = Math.min(
      Math.abs(c - origin),
      communityCount - Math.abs(c - origin),
    );
    const start = ringDist * 0.11;
    const local = Math.max(0, Math.min(1, (progress - start) / 0.42));
    clusterWithdrawShare[c] = local * local;
  }

  const population =
    base.panels.calm_count + base.panels.wavering_count + base.panels.withdrew_count;
  const perOrb = blockSize(population, base.point_count);
  const state: number[] = [];
  let calm = 0;
  let wavering = 0;
  let withdrew = 0;
  for (let i = 0; i < base.point_count; i += 1) {
    const cluster = ids[i] ?? 0;
    const share = clusterWithdrawShare[cluster] ?? 0;
    const jitter = ((base.seed + i * 17) % 100) / 100;
    let s = STATE_CALM;
    if (share > 0.55 + jitter * 0.15) {
      s = STATE_WITHDREW;
      withdrew += 1;
    } else if (share > 0.12 + jitter * 0.1) {
      s = STATE_WAVERING;
      wavering += 1;
    } else {
      calm += 1;
    }
    state.push(s);
  }

  const cash = Math.max(0.08, 1 - progress * 0.92);
  const tipping = Math.min(1, progress * 1.1);
  const ttf = cash < 0.2 ? 0 : Math.max(0, Math.round((1 - progress) * 180));

  return {
    ...base,
    positions_xy: base.positions_xy,
    balance_norm: base.balance_norm,
    uninsured_mask: base.uninsured_mask,
    point_count: base.point_count,
    t: elapsedSec,
    state,
    panels: {
      ...base.panels,
      rumor: rumorText || base.panels.rumor,
      calm_count: calm * perOrb,
      wavering_count: wavering * perOrb,
      withdrew_count: withdrew * perOrb,
      cash_gauge: cash,
      tipping_point: tipping,
      time_to_failure_ticks: ttf,
      state_ribbon: withdrew / base.point_count > 0.1 ? "crisis" : wavering > calm * 0.2 ? "wavering" : "calm",
    },
  };
}

export function fakeSpreadCycleSeconds(): number {
  return cycleSeconds();
}
