import type { CrowdFrame } from "./types";

function seededUnit(seed: number, index: number): number {
  const x = Math.sin(seed * 12.9898 + index * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

export type TransitionPlan = {
  from: CrowdFrame;
  to: CrowdFrame;
  durationMs: number;
  switchAt: Float32Array;
};

export function planTransition(from: CrowdFrame, to: CrowdFrame, durationMs: number): TransitionPlan {
  const switchAt = new Float32Array(from.point_count);
  for (let i = 0; i < from.point_count; i += 1) {
    const prev = from.state[i] ?? 0;
    const next = to.state[i] ?? 0;
    if (prev === next) {
      switchAt[i] = 0;
    } else {
      switchAt[i] = 0.08 + seededUnit(from.seed, i) * 0.82;
    }
  }
  return { from, to, durationMs, switchAt };
}

function blendCounts(from: CrowdFrame, to: CrowdFrame, alpha: number): CrowdFrame["panels"] {
  const p0 = from.panels;
  const p1 = to.panels;
  const mix = (a: number, b: number) => a + (b - a) * alpha;
  const calm = Math.round(mix(p0.calm_count, p1.calm_count));
  const wavering = Math.round(mix(p0.wavering_count, p1.wavering_count));
  const withdrew = Math.round(mix(p0.withdrew_count, p1.withdrew_count));
  const cash = mix(p0.cash_gauge, p1.cash_gauge);
  const tipping = mix(p0.tipping_point, p1.tipping_point);
  const ttf =
    p0.time_to_failure_ticks === null && p1.time_to_failure_ticks === null
      ? null
      : mix(p0.time_to_failure_ticks ?? 0, p1.time_to_failure_ticks ?? 0);
  const ribbon = alpha < 0.5 ? p0.state_ribbon : p1.state_ribbon;
  const rumor = alpha < 0.85 ? p1.rumor || p0.rumor : p1.rumor;
  return {
    ...p1,
    calm_count: calm,
    wavering_count: wavering,
    withdrew_count: withdrew,
    cash_gauge: cash,
    tipping_point: tipping,
    time_to_failure_ticks: ttf,
    state_ribbon: ribbon,
    rumor,
  };
}

function sameOrbLayout(from: CrowdFrame, to: CrowdFrame): boolean {
  return (
    from.point_count === to.point_count &&
    from.positions_xy.length === to.positions_xy.length &&
    from.positions_xy.length >= from.point_count * 2
  );
}

export function interpolateFrame(plan: TransitionPlan, elapsedMs: number): CrowdFrame {
  const alpha = Math.min(1, Math.max(0, elapsedMs / plan.durationMs));
  const count = Math.min(plan.from.point_count, plan.from.state.length, plan.to.state.length);
  const state = plan.from.state.slice();
  for (let i = 0; i < count; i += 1) {
    const prev = plan.from.state[i] ?? 0;
    const next = plan.to.state[i] ?? 0;
    const threshold = plan.switchAt[i] ?? 0;
    if (threshold <= 0) {
      state[i] = prev;
    } else if (alpha >= threshold) {
      state[i] = next;
    } else {
      state[i] = prev < next ? Math.min(prev + 1, next) : prev;
    }
  }
  const pinned = sameOrbLayout(plan.from, plan.to);
  return {
    ...(pinned ? plan.from : plan.to),
    positions_xy: pinned ? plan.from.positions_xy : plan.to.positions_xy,
    balance_norm: pinned ? plan.from.balance_norm : plan.to.balance_norm,
    uninsured_mask: pinned ? plan.from.uninsured_mask : plan.to.uninsured_mask,
    point_count: pinned ? plan.from.point_count : plan.to.point_count,
    seed: pinned ? plan.from.seed : plan.to.seed,
    bank_id: plan.to.bank_id ?? plan.from.bank_id,
    t: plan.from.t + (plan.to.t - plan.from.t) * alpha,
    state,
    panels: blendCounts(plan.from, plan.to, alpha),
  };
}
