import { useCallback, useRef, useState } from "react";

const API_BASE =
  (import.meta.env.VITE_STAMPEDE_API as string | undefined) ?? "http://127.0.0.1:8000";

export type PlaybackSpeed = 1 | 2 | 4 | 8;

const SPEED_STEPS: PlaybackSpeed[] = [1, 2, 4, 8];

function nearestSpeed(speed: number): PlaybackSpeed {
  return SPEED_STEPS.reduce((best, step) =>
    Math.abs(step - speed) < Math.abs(best - speed) ? step : best,
  );
}

export function usePlaybackControls(liveMode: boolean) {
  const [paused, setPaused] = useState(true);
  const [speed, setSpeed] = useState<PlaybackSpeed>(1);
  const controlAt = useRef(0);

  const sendControl = useCallback(
    async (action: string, extra?: Record<string, unknown>) => {
      controlAt.current = Date.now();
      if (!liveMode) {
        return;
      }
      await fetch(`${API_BASE}/api/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...extra }),
      });
    },
    [liveMode],
  );

  const togglePause = useCallback(() => {
    setPaused((p) => {
      const next = !p;
      if (liveMode) {
        void sendControl(next ? "pause" : "resume");
      }
      return next;
    });
  }, [liveMode, sendControl]);

  const changeSpeed = useCallback(
    (next: PlaybackSpeed) => {
      setSpeed(next);
      if (liveMode) {
        void sendControl("set_speed", { speed: next });
      }
    },
    [liveMode, sendControl],
  );

  const reset = useCallback(() => {
    setPaused(true);
    setSpeed(1);
    if (liveMode) {
      void sendControl("reset");
    }
  }, [liveMode, sendControl]);

  const skipToOutcome = useCallback(() => {
    setPaused(true);
    if (liveMode) {
      void sendControl("skip_to_outcome");
    }
  }, [liveMode, sendControl]);

  const resume = useCallback(() => {
    setPaused(false);
    if (liveMode) {
      void sendControl("resume");
    }
  }, [liveMode, sendControl]);

  const applyServer = useCallback((nextPaused: boolean, nextSpeed: number) => {
    if (Date.now() - controlAt.current < 800) return;
    setPaused(nextPaused);
    setSpeed(nearestSpeed(nextSpeed));
  }, []);

  return {
    paused,
    speed,
    togglePause,
    changeSpeed,
    reset,
    skipToOutcome,
    resume,
    setPaused,
    applyServer,
  };
}
