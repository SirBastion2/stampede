import type { CrowdFrame } from "./types";

export type CrisisPhase = "calm" | "rumor" | "run" | "response" | "outcome";

const PHASES: CrisisPhase[] = ["calm", "rumor", "run", "response", "outcome"];

export function crisisPhase(frame: CrowdFrame, crisisMinutes?: number): CrisisPhase {
  const p = frame.panels;
  const total = Math.max(1, p.calm_count + p.wavering_count + p.withdrew_count);
  const withdrewShare = p.withdrew_count / total;
  const waveringShare = p.wavering_count / total;
  const failed = p.time_to_failure_ticks === 0 || p.cash_gauge < 0.15;
  const horizon = crisisMinutes && crisisMinutes > 0 ? crisisMinutes : 0;
  const progress = horizon > 0 ? frame.t / horizon : 0;
  // The cash buffer can empty while most depositors are still calm. Outcome is
  // the end of the registry crisis window, not that first empty reading.
  if (failed && horizon > 0 && progress >= 0.9) {
    return "outcome";
  }
  if (failed && horizon === 0 && waveringShare >= 0.45 && p.tipping_point >= 0.95) {
    return "outcome";
  }
  if (
    p.rumor &&
    (failed || withdrewShare >= 0.002 || waveringShare >= 0.12 || p.state_ribbon === "crisis")
  ) {
    return "run";
  }
  if (p.rumor && (p.tipping_point >= 0.35 || waveringShare >= 0.08)) {
    return "response";
  }
  if (p.rumor || p.tipping_point >= 0.2) {
    return "rumor";
  }
  return "calm";
}

export function phaseList(): CrisisPhase[] {
  return PHASES;
}

export function formatCashBillions(cashGauge: number): string {
  const billions = Math.max(0, cashGauge) * 5;
  if (billions >= 10) return `$${billions.toFixed(0)}B`;
  return `$${billions.toFixed(1)}B`;
}

export function formatTimeToFailure(ticks: number | null): { main: string; sub: string } {
  if (ticks === null) {
    return { main: "—", sub: "" };
  }
  const minutesPerTick = 5;
  const totalMinutes = Math.max(0, ticks * minutesPerTick);
  const hours = Math.floor(totalMinutes / 60);
  const mins = Math.round(totalMinutes % 60);
  const main = hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  let sub = "";
  if (ticks === 0) {
    sub = "Past the tipping point";
  } else if (ticks < 120) {
    sub = "Window closing";
  }
  return { main, sub };
}
