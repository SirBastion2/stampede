import { useEffect, useMemo, useState } from "react";
import { CrowdViewport } from "./CrowdViewport";
import { MarketEmbed } from "./MarketEmbed";
import { blockSize } from "./orbStyle";
import { crisisPhase, formatCashBillions, phaseList } from "./phase";
import { PlaybackBar } from "./PlaybackBar";
import { SidePanels } from "./SidePanels";
import { useCrowdFeed } from "./useCrowdFeed";

function bankLabel(bankId: string | null): string {
  if (!bankId) return "Kestrel Bank · fictional";
  if (bankId.includes("fvb") || bankId.includes("kestrel")) {
    return "Kestrel Bank · fictional";
  }
  return `${bankId} · fictional`;
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const stored = window.localStorage.getItem("stampede-motion");
    const media = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const on = stored === "reduce" || (stored !== "full" && media);
    setReduced(on);
    document.documentElement.dataset.motion = on ? "reduce" : "full";
  }, []);
  const toggle = () => {
    setReduced((current) => {
      const next = !current;
      window.localStorage.setItem("stampede-motion", next ? "reduce" : "full");
      document.documentElement.dataset.motion = next ? "reduce" : "full";
      return next;
    });
  };
  return { reduced, toggle };
}

export function App() {
  const {
    frame,
    connection,
    crisisMinutes,
    demoPopulation,
    poster,
    setPoster,
    draftRumor,
    setDraftRumor,
    submitRumor,
    moderationError,
    rumorInterpretation,
    rumorBreakdown,
    rumorJudgment,
    setReducedMotion,
    playback,
  } = useCrowdFeed();
  const { reduced, toggle } = useReducedMotion();
  useEffect(() => {
    setReducedMotion(reduced);
  }, [reduced, setReducedMotion]);

  const phase = useMemo(
    () => crisisPhase(frame, crisisMinutes || undefined),
    [frame, crisisMinutes],
  );
  const cashLabel = useMemo(
    () => formatCashBillions(frame.panels.cash_gauge),
    [frame.panels.cash_gauge],
  );
  const people = frame.panels.calm_count + frame.panels.wavering_count + frame.panels.withdrew_count;
  const peoplePerOrb = useMemo(
    () => blockSize(Math.max(1, people), Math.max(1, frame.point_count)),
    [people, frame.point_count],
  );
  const reconnecting = connection === "connecting" || connection === "reconnecting";

  return (
    <div className="monitor-shell" data-feed-mode={connection} data-phase={phase}>
      <header className="monitor-header">
        <div className="monitor-title-row">
          <h1>Left monitor: the crowd</h1>
          <span className="bank-badge">{bankLabel(frame.bank_id)}</span>
        </div>
        <p className="demo-hook">
          For bank risk teams and supervisors. A fictional 2023-style run. Kestrel Bank is fictional.
        </p>
        <nav className="phase-pills" aria-label="Crisis phase">
          {phaseList().map((id) => (
            <span
              key={id}
              className={`phase-pill ${id === phase ? "phase-pill-active" : ""}`}
              aria-current={id === phase ? "step" : undefined}
            >
              {id.charAt(0).toUpperCase() + id.slice(1)}
            </span>
          ))}
        </nav>
        <PlaybackBar
          paused={playback.paused}
          speed={playback.speed}
          reducedMotion={reduced}
          onTogglePause={playback.togglePause}
          onSpeed={playback.changeSpeed}
          onReset={playback.reset}
          onSkip={playback.skipToOutcome}
          onToggleMotion={toggle}
        />
      </header>

      {reconnecting ? (
        <p className="reconnect-banner" role="status">
          {connection === "connecting"
            ? "Connecting to the simulation…"
            : "Reconnecting to the simulation…"}
        </p>
      ) : null}

      <section className="crowd-stage card">
        <div className="crowd-canvas-wrap">
          <CrowdViewport frame={frame} reducedMotion={reduced} />
        </div>
        <div className="crowd-legend">
          <span>
            <i className="leg calm" /> calm {frame.panels.calm_count.toLocaleString()}
          </span>
          <span>
            <i className="leg wavering" /> wavering {frame.panels.wavering_count.toLocaleString()}
          </span>
          <span>
            <i className="leg withdrew" /> withdrew {frame.panels.withdrew_count.toLocaleString()}
          </span>
          <span className="leg-note">big dot = uninsured</span>
          <span className="leg-note">
            1 orb = {peoplePerOrb.toLocaleString()} people · {people.toLocaleString()} depositors
          </span>
        </div>
      </section>

      <SidePanels
        frame={frame}
        poster={poster}
        onPosterChange={setPoster}
        draftRumor={draftRumor}
        onDraftRumorChange={setDraftRumor}
        phase={phase}
        cashLabel={cashLabel}
        onRumorSubmit={submitRumor}
        moderationError={moderationError}
        rumorInterpretation={rumorInterpretation}
        rumorBreakdown={rumorBreakdown}
        rumorJudgment={rumorJudgment}
      />

      <MarketEmbed />

      <footer className="demo-foot">
        <p>Fictional simulation. Not a forecast and not a live market.</p>
        <p>
          This run draws {frame.point_count.toLocaleString()} orbs for {people.toLocaleString()} depositors
          {demoPopulation > 0
            ? ` (registry pitch population ${demoPopulation.toLocaleString()})`
            : ""}
          . CPU fake engine, offline. Calibration against 2023 runs is not fitted in this demo.
        </p>
      </footer>
    </div>
  );
}
