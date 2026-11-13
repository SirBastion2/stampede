import type { MarketScreenFrame } from "../types";

type Props = { frame: MarketScreenFrame };

export function DepthBars({ frame }: Props) {
  if (frame.depth.length === 0) {
    return null;
  }
  const mark = frame.prices[frame.prices.length - 1]?.[1] ?? 0;
  let bidSize = 0;
  let askSize = 0;
  for (const level of frame.depth) {
    if (level.price <= mark) bidSize += level.size;
    else askSize += level.size;
  }
  const total = bidSize + askSize || 1;
  const buyerPct = Math.round((bidSize / total) * 100);
  const sellerPct = 100 - buyerPct;

  return (
    <section className="panel depth-panel" aria-label="Order book depth">
      <h2>Order book depth</h2>
      <div className="depth-combo" role="img" aria-label={`Buyers ${buyerPct} percent, sellers ${sellerPct} percent`}>
        <div className="depth-buy" style={{ width: `${buyerPct}%` }} />
        <div className="depth-sell" style={{ width: `${sellerPct}%` }} />
      </div>
      <div className="depth-labels">
        <span className="buy-label">Buyers {buyerPct}%</span>
        <span className="sell-label">Sellers {sellerPct}%</span>
      </div>
    </section>
  );
}
