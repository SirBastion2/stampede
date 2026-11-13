import { CrisisStatements } from "../../market/src/components/CrisisStatements";
import { DepthBars } from "../../market/src/components/DepthBars";
import { HaltBanner } from "../../market/src/components/HaltBanner";
import { PriceChart, priceHeadline } from "../../market/src/components/PriceChart";
import { useMarketFeed } from "../../market/src/hooks/useMarketFeed";
import "./marketEmbed.css";

export function MarketEmbed() {
  const { frame, mode, error } = useMarketFeed();
  const headline = frame ? priceHeadline(frame) : null;

  return (
    <section className="market-region" aria-label="Right monitor: the market">
      <header className="market-embed-header">
        <h2>Right monitor: the market</h2>
        {headline && headline.change ? (
          <p className="ticker-lockup">
            <span className="ticker">KSTL</span>
            <span className="ticker-price">${headline.price}</span>
            <span className={`ticker-change ${headline.down ? "down" : "up"}`}>
              {headline.down ? "↓" : "↑"} {headline.change}
              <span className="sr-only">{headline.down ? " down" : " up"}</span>
            </span>
          </p>
        ) : (
          <p className="ticker-lockup">KSTL waiting for the first trade</p>
        )}
      </header>
      {mode !== "live" && mode !== "fake" ? (
        <p className="reconnect-banner" role="status">
          {error ?? "Connecting to the market feed…"}
        </p>
      ) : null}
      {mode === "live" && error ? (
        <p className="reconnect-banner" role="status">
          {error}
        </p>
      ) : null}
      {!frame ? (
        <p className="market-empty">Market chart appears when the feed connects.</p>
      ) : (
        <>
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
        </>
      )}
    </section>
  );
}
