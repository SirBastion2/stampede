import { formatTimeToFailure } from "./phase";
import type { CrowdFrame } from "./types";
import type { CredibilityBreakdown, PosterId, RumorJudgment } from "./useCrowdFeed";
import { POSTERS } from "./useCrowdFeed";
import type { CrisisPhase } from "./phase";

interface SidePanelsProps {
  frame: CrowdFrame;
  poster: PosterId;
  onPosterChange: (poster: PosterId) => void;
  draftRumor: string;
  onDraftRumorChange: (text: string) => void;
  phase: CrisisPhase;
  cashLabel: string;
  onRumorSubmit?: () => void;
  moderationError?: string | null;
  rumorInterpretation?: string | null;
  rumorBreakdown?: CredibilityBreakdown | null;
  rumorJudgment?: RumorJudgment | null;
}

// One label per role id in POSTERS (a Python test checks every id has one).
const POSTER_LABELS: Record<PosterId, string> = {
  "ui-user": "Regular user",
  "group-chat-anon": "Anonymous group chat",
  vip: "VIP / celebrity, huge following",
  "treasury-desk": "Treasury desk",
  "short-seller": "Short-seller research firm",
  journalist: "Financial journalist / wire",
  "founder-circle": "Founder circle (depositors)",
  "bank-ceo": "The bank's CEO / CFO",
  "major-shareholder": "Top shareholder",
  "vc-partner": "VC, 100k followers",
  regulator: "Regulator / central bank",
};

function posterChipLabel(poster: string | null | undefined): string {
  if (!poster) return "Unknown poster";
  return POSTER_LABELS[poster as PosterId] ?? poster;
}

function fmt(value: number): string {
  return value.toFixed(2);
}

function signed(value: number): string {
  return `${value < 0 ? "−" : "+"} ${fmt(Math.abs(value))}`;
}

/** "Bank CEO 0.85 + specificity 0.05 = 0.90 (cap)" — the code path, not a model guess. */
function breakdownSentence(b: CredibilityBreakdown): string {
  const parts = [`${posterChipLabel(b.role)} ${fmt(b.role_base)}`];
  parts.push(`specificity ${signed(b.specificity_bonus)}`);
  if (b.model_shift !== 0) parts.push(`model ${signed(b.model_shift)}`);
  const tail = b.clamped ? (b.raw > b.cap ? " (capped)" : " (floored)") : "";
  const judge = b.judge_source === "model" ? " · judged by model" : "";
  return `Credibility: ${parts.join(" ")} = ${fmt(b.credibility)}${tail}${judge}`;
}

/** "Judge: liquidity claim · call to action · plausibility 0.90 (model)". */
function judgmentSentence(j: RumorJudgment): string {
  const parts = [`${j.claim_type} claim`];
  if (j.call_to_action) parts.push("call to action");
  if (j.source === "model") parts.push(`plausibility ${fmt(j.plausibility)}`);
  const why = j.source === "model" ? "model" : j.note ? `lexicon, ${j.note}` : "lexicon";
  return `Judge: ${parts.join(" · ")} (${why})`;
}

export function SidePanels({
  frame,
  poster,
  onPosterChange,
  draftRumor,
  onDraftRumorChange,
  phase,
  cashLabel,
  onRumorSubmit,
  moderationError,
  rumorInterpretation,
  rumorBreakdown,
  rumorJudgment,
}: SidePanelsProps) {
  const panels = frame.panels;
  const cashPct = Math.round(Math.min(panels.cash_gauge, 1) * 100);
  const ttf = formatTimeToFailure(panels.time_to_failure_ticks);
  const cashEmpty = panels.cash_gauge < 0.15;
  const cashFalling = cashEmpty || phase === "run" || phase === "outcome";
  const posted = panels.rumor.trim();
  const timeSub =
    phase === "outcome"
      ? "Past the tipping point"
      : panels.time_to_failure_ticks === 0
        ? "Cash buffer empty"
        : ttf.sub;

  return (
    <aside className="side-panels">
      <div className="stat-row">
        <section className="panel stat-card">
          <h2>Cash on hand</h2>
          <p className={`stat-value ${cashFalling ? "stat-value-down" : "stat-value-steady"}`}>
            {cashLabel}
            {cashFalling ? (
              <span className="trend-down">
                {" "}
                ↓<span className="sr-only"> down</span>
              </span>
            ) : (
              <span className="sr-only"> steady</span>
            )}
          </p>
          <div
            className={`gauge-track ${cashFalling ? "gauge-track-danger" : ""}`}
            role="img"
            aria-label={`Operating cash buffer ${cashPct} percent`}
          >
            <div className="gauge-fill" style={{ width: `${Math.min(cashPct, 100)}%` }} />
          </div>
          {cashEmpty ? <p className="stat-sub">Operating cash buffer is empty</p> : null}
        </section>
        <section className="panel stat-card">
          <h2>Time to failure</h2>
          <p className="stat-value stat-value-down">{ttf.main}</p>
          {timeSub ? <p className="stat-sub stat-sub-warn">{timeSub}</p> : null}
        </section>
      </div>

      <section className="panel rumor-box">
        {!posted ? (
          <p className="rumor-hint" id="rumor-hint">
            Type a rumor about Kestrel Bank, then press Enter.
          </p>
        ) : (
          <blockquote className="rumor-quote">&ldquo;{posted}&rdquo;</blockquote>
        )}
        <div className="rumor-compose">
          <label className="sr-only" htmlFor="rumor-draft">
            Rumor
          </label>
          <input
            id="rumor-draft"
            value={draftRumor}
            onChange={(event) => onDraftRumorChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                onRumorSubmit?.();
              }
            }}
            placeholder="I heard they lost billions on bonds"
            aria-describedby={posted ? undefined : "rumor-hint"}
            autoComplete="off"
          />
          <button type="button" className="post-btn" onClick={onRumorSubmit}>
            Post rumor
          </button>
          <label className="sr-only" htmlFor="poster-select">
            Who posts it
          </label>
          <select
            id="poster-select"
            className="poster-chip poster-select"
            value={poster}
            onChange={(event) => onPosterChange(event.target.value as PosterId)}
            aria-label="Who posts it"
          >
            {POSTERS.map((id) => (
              <option key={id} value={id}>
                {posterChipLabel(id)}
              </option>
            ))}
          </select>
        </div>
        {rumorInterpretation ? (
          <p className="rumor-interpretation">{rumorInterpretation}</p>
        ) : null}
        {rumorBreakdown ? (
          <p className="rumor-interpretation rumor-breakdown">{breakdownSentence(rumorBreakdown)}</p>
        ) : null}
        {rumorJudgment ? (
          <p className="rumor-interpretation rumor-judgment">{judgmentSentence(rumorJudgment)}</p>
        ) : null}
        {moderationError ? (
          <p className="rumor-error" role="alert">
            {moderationError}
          </p>
        ) : null}
      </section>
    </aside>
  );
}
