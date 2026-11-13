import { useEffect, useRef } from "react";
import type { MarketScreenFrame } from "../types";
import { statementLabel } from "./CrisisStatements";

type Props = { frame: MarketScreenFrame };

function markerLabel(label: string): string {
  const lower = label.toLowerCase();
  if (lower.includes("halt")) return "Halt";
  if (lower.includes("rumor")) return "Rumor posted";
  if (lower.includes("true")) return "True statement";
  return statementLabel(label);
}

export function priceHeadline(frame: MarketScreenFrame): {
  price: string;
  change: string;
  down: boolean;
} {
  const last = frame.prices[frame.prices.length - 1];
  const first = frame.prices[0];
  const pct = last && first && first[1] !== 0 ? ((last[1] - first[1]) / first[1]) * 100 : 0;
  const down = pct < 0;
  if (!last || frame.prices.length < 2) {
    return { price: "—", change: "", down: false };
  }
  return {
    price: last[1].toFixed(2),
    change: `${Math.abs(pct).toFixed(1)}%`,
    down,
  };
}

export function PriceChart({ frame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || frame.prices.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width < 2 || height < 2) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const pad = { l: 56, r: 16, t: 48, b: 32 };
    const plotW = width - pad.l - pad.r;
    const plotH = height - pad.t - pad.b;
    const xs = frame.prices.map((p) => p[0]);
    const ys = frame.prices.map((p) => p[1]);
    const xMin = xs[0];
    const xMax = xs[xs.length - 1];
    const yMax = Math.max(...ys, 1) * 1.05;
    const yMin = 0;

    const xScale = (t: number) => pad.l + ((t - xMin) / (xMax - xMin || 1)) * plotW;
    const yScale = (v: number) => pad.t + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (let i = 1; i < frame.prices.length; i += 1) {
      const [t0, v0] = frame.prices[i - 1];
      const [t1, v1] = frame.prices[i];
      const falling = v1 < v0 - 0.01;
      const rising = v1 > v0 + 0.01;
      ctx.strokeStyle = falling ? "#ef6b6b" : rising ? "#3ecf8e" : "#d6d3d1";
      ctx.lineWidth = 2.25;
      ctx.beginPath();
      ctx.moveTo(xScale(t0), yScale(v0));
      ctx.lineTo(xScale(t1), yScale(v1));
      ctx.stroke();
    }

    for (const marker of frame.markers) {
      const x = Math.min(pad.l + plotW - 8, Math.max(pad.l + 8, xScale(marker.t)));
      ctx.save();
      ctx.strokeStyle = "rgba(231, 229, 228, 0.7)";
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + plotH);
      ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = "#e7e5e4";
    ctx.font = "12px Segoe UI, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`$${yMax.toFixed(0)}`, pad.l - 6, yScale(yMax) + 4);
    ctx.fillText("$0", pad.l - 6, yScale(yMin));
    ctx.textAlign = "left";
    ctx.fillText(`${Math.round(xMin)} min`, pad.l, pad.t + plotH + 16);
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(xMax)} min`, pad.l + plotW, pad.t + plotH + 16);
    ctx.fillText("Simulated minutes", pad.l + plotW / 2, height - 4);
    };
    draw();
    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [frame]);

  if (frame.prices.length < 2) {
    return (
      <section className="panel chart-panel" aria-label="Stock price chart">
        <p className="market-empty">Price chart fills in after the first trades.</p>
      </section>
    );
  }

  return (
    <section className="panel chart-panel" aria-label="Stock price in dollars versus simulated minutes">
      {frame.markers.length > 0 ? (
        <ul className="marker-legend">
          {frame.markers.map((marker, index) => (
            <li key={`${marker.t}-${marker.label}-${index}`}>{markerLabel(marker.label)}</li>
          ))}
        </ul>
      ) : null}
      <canvas ref={canvasRef} className="price-canvas" />
    </section>
  );
}
