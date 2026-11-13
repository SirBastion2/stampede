import { FAKE_FRAME } from "./fakeFrame";
import type { MarketScreenFrame } from "./types";

const CYCLE_SEC = 105;

export function fakeMarketFrame(elapsedSec: number): MarketScreenFrame {
  const phase = Math.min(1, (elapsedSec % CYCLE_SEC) / CYCLE_SEC);
  const base = FAKE_FRAME;
  const lastT = base.prices[base.prices.length - 1][0];
  const t = phase * lastT;
  const prices: [number, number][] = [];
  for (let i = 0; i < base.prices.length; i += 1) {
    const [pt, pv] = base.prices[i];
    if (pt <= t) {
      const next = base.prices[i + 1];
      if (next && next[0] > t) {
        const alpha = (t - pt) / (next[0] - pt);
        const v = pv + (next[1] - pv) * alpha;
        prices.push([t, v]);
        break;
      }
      prices.push([pt, pv]);
    }
  }
  if (prices.length < 2) {
    prices.push([0, base.prices[0][1]], [t || 1, base.prices[1][1]]);
  }
  const volScale = 1 + phase * 2.5;
  const volumes = base.volumes.slice(0, prices.length).map((v, i) =>
    Math.round(v * (1 + phase * (i / prices.length))),
  );
  const halt = phase > 0.72 ? { reason: "volatility", until_t: lastT } : null;
  return {
    ...base,
    t,
    prices,
    volumes,
    halt,
    depth: base.depth.map((row, i) => ({
      price: row.price * (1 - phase * 0.04 * (i / base.depth.length)),
      size: Math.round(row.size * volScale),
    })),
  };
}
