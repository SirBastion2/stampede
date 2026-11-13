import type { MarketScreenFrame } from "./types";

/** Static run-phase frame for `?fake=1`. Styled to the monitor comp. */
function demoPricePath(): [number, number][] {
  const points: [number, number][] = [];
  const steps = 48;
  for (let i = 0; i <= steps; i += 1) {
    const t = (i / steps) * 10;
    let price: number;
    if (t <= 3.2) {
      price = 30 - t * 0.1;
    } else if (t <= 7.2) {
      const u = (t - 3.2) / 4;
      price = 29.68 - (1 - Math.cos(u * Math.PI)) * 0.5 * 13.3;
    } else {
      const u = (t - 7.2) / (10 - 7.2);
      price = 16.38 + u * (18.42 - 16.38);
    }
    points.push([Math.round(t * 100) / 100, Math.round(price * 100) / 100]);
  }
  points[0][1] = 30;
  points[points.length - 1][1] = 18.42;
  return points;
}

const prices = demoPricePath();

export const FAKE_FRAME: MarketScreenFrame = {
  module: "M30",
  t: 10,
  bank_id: "bank-kestrel",
  prices,
  volumes: prices.map(() => 12000),
  depth: [
    { price: 18.1, size: 140 },
    { price: 18.3, size: 80 },
    { price: 18.6, size: 200 },
    { price: 18.9, size: 220 },
    { price: 19.3, size: 180 },
    { price: 19.8, size: 180 },
  ],
  halt: null,
  markers: [
    { t: 3.2, kind: "statement", label: "Rumor posted" },
    { t: 7.2, kind: "statement", label: "True statement" },
  ],
  statements: [
    {
      stmt_id: "ceo-video-liquidity-fed-line",
      channel: "video",
      verdict: "accepted",
      reason: "",
    },
    {
      stmt_id: "personal-text-uninsured",
      channel: "sms",
      verdict: "accepted",
      reason: "",
    },
    {
      stmt_id: "generic-well-capitalized",
      channel: "press",
      verdict: "accepted",
      reason: "",
    },
    {
      stmt_id: "all-deposits-guaranteed",
      channel: "x_post",
      verdict: "rejected",
      reason: "false",
    },
  ],
  futures: Array.from({ length: 16 }, (_, index) => ({
    future_id: `f${String(index).padStart(2, "0")}`,
    failure_probability: 0.12 + index * 0.035,
    price_low: 16 + index * 0.08,
    price_high: 22 - index * 0.04,
  })),
  headline_messages: 32,
  headline_futures: 32,
};
