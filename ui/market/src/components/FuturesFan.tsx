import type { FutureFanBar } from "../types";

type Props = { futures: FutureFanBar[] };

export function FuturesFan({ futures }: Props) {
  const sorted = [...futures].sort((a, b) => a.failure_probability - b.failure_probability);
  const maxP = Math.max(...sorted.map((f) => f.failure_probability), 0.01);
  return (
    <section className="panel futures-panel" aria-label="Futures fan">
      <h2>Futures fan ({sorted.length})</h2>
      <div className="futures-grid">
        {sorted.map((future) => (
          <div key={future.future_id} className="future-bar" title={future.future_id}>
            <div
              className="future-fill"
              style={{ height: `${(future.failure_probability / maxP) * 100}%` }}
            />
            <span className="future-label">{future.future_id}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
