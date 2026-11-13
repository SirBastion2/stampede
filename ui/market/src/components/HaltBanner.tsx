import type { MarketScreenFrame } from "../types";

type Props = { frame: MarketScreenFrame };

export function HaltBanner({ frame }: Props) {
  if (!frame.halt) {
    return null;
  }
  const reason = String(frame.halt.reason ?? "halt");
  const duration = frame.halt.duration;
  return (
    <div className="halt-banner" role="status">
      <strong>Trading halted</strong>
      <span>{reason}</span>
      {duration !== undefined && <span>{duration} min</span>}
    </div>
  );
}
