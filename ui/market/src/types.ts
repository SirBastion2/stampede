export type DepthLevel = { price: number; size: number };

export type ChartMarker = { t: number; kind: "halt" | "statement"; label: string };

export type CrisisStatementRow = {
  stmt_id: string;
  channel: string;
  verdict: "accepted" | "rejected" | "pending";
  reason: string;
};

export type FutureFanBar = {
  future_id: string;
  failure_probability: number;
  price_low: number | null;
  price_high: number | null;
};

export type MarketScreenFrame = {
  module: "M30";
  t: number;
  bank_id: string | null;
  prices: [number, number][];
  volumes: number[];
  depth: DepthLevel[];
  halt: Record<string, string | number> | null;
  markers: ChartMarker[];
  statements: CrisisStatementRow[];
  futures: FutureFanBar[];
  /** Demo headline when the visible rows are a sample of the crisis room. */
  headline_messages?: number;
  headline_futures?: number;
};
