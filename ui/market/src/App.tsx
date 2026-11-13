import { CrisisStatements } from "./components/CrisisStatements";
import { DepthBars } from "./components/DepthBars";
import { HaltBanner } from "./components/HaltBanner";
import { PriceChart, priceHeadline } from "./components/PriceChart";
import { useMarketFeed } from "./hooks/useMarketFeed";

export function App() {
  const { frame, mode, error } = useMarketFeed();

  if (!frame) {
    return (
      <main className="monitor-shell">
        <p className="connecting">Connecting to market feed…</p>
      </main>
    );
  }

  const headline = priceHeadline(frame);

  return (
    <main className="monitor-shell" data-feed-mode={mode}>
      <header className="monitor-header">
        <h1>Right monitor: the market</h1>
        {headline.change ? (
          <p className="ticker-lockup">
            <span className="ticker">KSTL</span>
            <span className="ticker-price">${headline.price}</span>
            <span className={`ticker-change ${headline.down ? "down" : "up"}`}>
              {headline.down ? "↓" : "↑"} {headline.change}
            </span>
          </p>
        ) : (
          <p className="ticker-lockup">KSTL waiting for the first trade</p>
        )}
        {error ? <p className="feed-error">{error}</p> : null}
      </header>
      <HaltBanner frame={frame} />
      <div className="market-stack">
        <PriceChart frame={frame} />
        <DepthBars frame={frame} />
        <CrisisStatements
          rows={frame.statements}
          futuresCount={frame.headline_futures ?? frame.futures.length}
          messageCount={frame.headline_messages ?? frame.statements.length}
        />
      </div>
    </main>
  );
}
