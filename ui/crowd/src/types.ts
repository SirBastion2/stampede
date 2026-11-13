export type StateRibbon = "calm" | "wavering" | "crisis";

export interface CrowdPanels {
  cash_gauge: number;
  time_to_failure_ticks: number | null;
  tipping_point: number;
  rumor: string;
  state_ribbon: StateRibbon;
  calm_count: number;
  wavering_count: number;
  withdrew_count: number;
}

export interface CrowdFrame {
  t: number;
  seed: number;
  bank_id: string | null;
  point_count: number;
  positions_xy: number[];
  state: number[];
  balance_norm: number[];
  uninsured_mask: boolean[];
  panels: CrowdPanels;
}
