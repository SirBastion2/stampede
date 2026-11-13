import type { PlaybackSpeed } from "./usePlaybackControls";

type Props = {
  paused: boolean;
  speed: PlaybackSpeed;
  reducedMotion: boolean;
  onTogglePause: () => void;
  onSpeed: (speed: PlaybackSpeed) => void;
  onReset: () => void;
  onSkip: () => void;
  onToggleMotion: () => void;
};

const SPEEDS: PlaybackSpeed[] = [1, 2, 4, 8];

export function PlaybackBar({
  paused,
  speed,
  reducedMotion,
  onTogglePause,
  onSpeed,
  onReset,
  onSkip,
  onToggleMotion,
}: Props) {
  return (
    <div className="playback-bar" aria-label="Simulation playback">
      <button type="button" className="playback-btn" onClick={onTogglePause}>
        {paused ? "Resume" : "Pause"}
      </button>
      <span className="playback-label">Speed</span>
      {SPEEDS.map((step) => (
        <button
          key={step}
          type="button"
          className={`playback-btn ${speed === step ? "playback-btn-active" : ""}`}
          onClick={() => onSpeed(step)}
        >
          {step}×
        </button>
      ))}
      <button type="button" className="playback-btn" onClick={onReset}>
        Reset
      </button>
      <button type="button" className="playback-btn playback-btn-skip" onClick={onSkip}>
        Skip to outcome
      </button>
      <button
        type="button"
        className={`playback-btn ${reducedMotion ? "playback-btn-active" : ""}`}
        onClick={onToggleMotion}
        aria-pressed={reducedMotion}
      >
        {reducedMotion ? "Motion reduced" : "Reduce motion"}
      </button>
    </div>
  );
}
