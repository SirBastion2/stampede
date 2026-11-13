import { useCallback, useEffect, useRef, useState } from "react";
import fakeFrame from "./fixtures/fake_frame.json";
import { decodeCrowdFrame } from "./crowdCodec";
import { fakeSpreadFrame } from "./fakeSpreadPlayback";
import { interpolateFrame, planTransition, type TransitionPlan } from "./interpolate";
import { plainModerationError } from "./moderationCopy";
import type { CrowdFrame } from "./types";
import { usePlaybackControls } from "./usePlaybackControls";

// Poster roles, weakest to strongest base credibility. Ids must match the
// registry row `M29 fake poster credibility (ASSUMED)`; a Python test checks it.
const POSTERS = [
  "ui-user",
  "group-chat-anon",
  "vip",
  "treasury-desk",
  "short-seller",
  "journalist",
  "founder-circle",
  "bank-ceo",
  "major-shareholder",
  "vc-partner",
  "regulator",
] as const;

export type PosterId = (typeof POSTERS)[number];

/** Server-computed credibility parts (`credibility_breakdown` in the API). */
export interface CredibilityBreakdown {
  role: string | null;
  role_known: boolean;
  role_base: number;
  specificity_hits: number;
  specificity_bonus: number;
  model_shift: number;
  raw: number;
  floor: number;
  cap: number;
  credibility: number;
  clamped: boolean;
  specificity?: number;
  judge_source?: string;
}

/** Judge scores (`judgment` in the API): lexicon offline, model when live. */
export interface RumorJudgment {
  specificity: number;
  plausibility: number;
  claim_type: string;
  call_to_action: boolean;
  source: string;
  note: string;
}

export type FeedConnection = "connecting" | "live" | "reconnecting" | "fake";

const API_BASE =
  (import.meta.env.VITE_STAMPEDE_API as string | undefined) ?? "http://127.0.0.1:8000";

const FRAME_MS = 1000 / 30;

function calmPlaceholder(base: CrowdFrame): CrowdFrame {
  const people =
    base.panels.calm_count + base.panels.wavering_count + base.panels.withdrew_count;
  return {
    ...base,
    t: 0,
    state: base.state.map(() => 0),
    panels: {
      cash_gauge: 1,
      time_to_failure_ticks: null,
      tipping_point: 0,
      rumor: "",
      state_ribbon: "calm",
      calm_count: Math.max(1, people),
      wavering_count: 0,
      withdrew_count: 0,
    },
  };
}

export function useCrowdFeed() {
  const fakeMode = import.meta.env.VITE_CROWD_FAKE !== "0";
  const initial = fakeMode
    ? (fakeFrame as CrowdFrame)
    : calmPlaceholder(fakeFrame as CrowdFrame);
  const [wireFrame, setWireFrame] = useState<CrowdFrame>(initial);
  const [displayFrame, setDisplayFrame] = useState<CrowdFrame>(initial);
  const [poster, setPoster] = useState<PosterId>("vc-partner");
  const [draftRumor, setDraftRumor] = useState("");
  const [moderationError, setModerationError] = useState<string | null>(null);
  const [connection, setConnection] = useState<FeedConnection>(fakeMode ? "fake" : "connecting");
  const [crisisMinutes, setCrisisMinutes] = useState(0);
  const [demoPopulation, setDemoPopulation] = useState(0);
  const [fakePlaying, setFakePlaying] = useState(false);
  const [fakeRumor, setFakeRumor] = useState("");
  const [fakeEffectScale, setFakeEffectScale] = useState(0);
  const [rumorInterpretation, setRumorInterpretation] = useState<string | null>(null);
  const [rumorBreakdown, setRumorBreakdown] = useState<CredibilityBreakdown | null>(null);
  const [rumorJudgment, setRumorJudgment] = useState<RumorJudgment | null>(null);
  const fakeStartRef = useRef<number | null>(null);
  const fakePausedAtRef = useRef<number | null>(null);
  const fakeSpeedRef = useRef(1);

  const playback = usePlaybackControls(!fakeMode);
  const planRef = useRef<TransitionPlan | null>(null);
  const planStartRef = useRef(0);
  const reducedMotionRef = useRef(false);

  useEffect(() => {
    if (fakeMode) {
      setWireFrame(fakeFrame as CrowdFrame);
      setDisplayFrame(fakeFrame as CrowdFrame);
      setConnection("fake");
      return;
    }
    let stopped = false;
    let socket: WebSocket | null = null;
    let retry = 0;
    const connect = () => {
      if (stopped) return;
      const url = import.meta.env.VITE_CROWD_WS_URL || "ws://127.0.0.1:8000/ws/crowd";
      try {
        socket = new WebSocket(url);
      } catch {
        setConnection("reconnecting");
        window.setTimeout(connect, Math.min(4000, 400 * (retry + 1)));
        retry += 1;
        return;
      }
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        retry = 0;
        setConnection("live");
        setModerationError((current) =>
          current?.startsWith("Reconnecting") ? null : current,
        );
      };
      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          try {
            const payload = JSON.parse(event.data) as { ok?: boolean; reasons?: string[] };
            if (payload.ok === false) {
              setModerationError(plainModerationError(payload.reasons));
            }
          } catch {
            /* ignore non-json text frames */
          }
          return;
        }
        const next = decodeCrowdFrame(event.data as ArrayBuffer);
        setWireFrame((prev) => {
          const duration = reducedMotionRef.current ? 0 : FRAME_MS;
          planRef.current = planTransition(prev, next, duration);
          planStartRef.current = performance.now();
          return next;
        });
        setConnection("live");
      };
      socket.onclose = () => {
        if (stopped) return;
        setConnection("reconnecting");
        window.setTimeout(connect, Math.min(4000, 400 * (retry + 1)));
        retry += 1;
      };
    };
    connect();
    return () => {
      stopped = true;
      socket?.close();
    };
  }, [fakeMode]);

  useEffect(() => {
    if (fakeMode) return;
    let cancel = false;
    const pull = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) return;
        const body = (await response.json()) as {
          paused?: boolean;
          speed?: number;
          crisis_sim_minutes?: number;
          demo_population?: number;
        };
        if (cancel) return;
        playback.applyServer(Boolean(body.paused), Number(body.speed ?? 1));
        if (body.crisis_sim_minutes) setCrisisMinutes(body.crisis_sim_minutes);
        if (body.demo_population) setDemoPopulation(body.demo_population);
      } catch {
        /* socket handler owns the reconnecting banner */
      }
    };
    void pull();
    const id = window.setInterval(() => void pull(), 1000);
    return () => {
      cancel = true;
      window.clearInterval(id);
    };
  }, [fakeMode, playback.applyServer]);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      if (fakeMode) {
        if (fakePlaying && fakeStartRef.current !== null) {
          const elapsed =
            ((performance.now() - fakeStartRef.current) / 1000) * fakeSpeedRef.current;
          const spread = fakeSpreadFrame(
            fakeFrame as CrowdFrame,
            elapsed,
            fakeRumor,
            fakeEffectScale,
          );
          setDisplayFrame(spread);
          setWireFrame(spread);
        }
      } else {
        const plan = planRef.current;
        if (plan) {
          const elapsed = performance.now() - planStartRef.current;
          setDisplayFrame(interpolateFrame(plan, elapsed));
        } else {
          setDisplayFrame(wireFrame);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [fakeMode, fakePlaying, fakeRumor, fakeEffectScale, wireFrame]);

  const submitRumor = useCallback(async () => {
    const text = draftRumor.trim();
    if (!text) {
      setModerationError("Type a rumor before posting.");
      return;
    }
    setModerationError(null);
    setRumorInterpretation(null);
    setRumorBreakdown(null);
    setRumorJudgment(null);
    const interpret = async (): Promise<{
      ok: boolean;
      read_as?: string;
      affects_simulation?: boolean;
      effect_size?: number;
      reasons?: string[];
      credibility_breakdown?: CredibilityBreakdown | null;
      judgment?: RumorJudgment | null;
    }> => {
      const response = await fetch(`${API_BASE}/api/rumor/interpret`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, poster }),
      });
      return (await response.json()) as {
        ok: boolean;
        read_as?: string;
        affects_simulation?: boolean;
        effect_size?: number;
        reasons?: string[];
        credibility_breakdown?: CredibilityBreakdown | null;
        judgment?: RumorJudgment | null;
      };
    };
    if (fakeMode) {
      try {
        const payload = await interpret();
        if (!payload.ok) {
          setModerationError(
            plainModerationError(payload.reasons, "That rumor was blocked. Try a fictional bank such as Kestrel Bank."),
          );
          return;
        }
        const readAs = payload.read_as ?? "";
        setRumorInterpretation(readAs);
        setRumorBreakdown(payload.credibility_breakdown ?? null);
        setRumorJudgment(payload.judgment ?? null);
        const scale = payload.affects_simulation
          ? Number(payload.effect_size ?? 0)
          : 0;
        setFakeRumor(text);
        setFakeEffectScale(scale);
        if (scale > 0) {
          fakeStartRef.current = performance.now();
          setFakePlaying(true);
          playback.setPaused(false);
          playback.resume();
        } else {
          setFakePlaying(false);
          fakeStartRef.current = null;
          setWireFrame(fakeFrame as CrowdFrame);
          setDisplayFrame({
            ...(fakeFrame as CrowdFrame),
            panels: {
              ...(fakeFrame as CrowdFrame).panels,
              rumor: text,
            },
          });
        }
        setDraftRumor("");
      } catch {
        setModerationError("Can't reach the simulation API to interpret that rumor.");
      }
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/rumor`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, poster, bank_id: wireFrame.bank_id }),
      });
      const payload = (await response.json()) as {
        ok?: boolean;
        reasons?: string[];
        read_as?: string;
        affects_simulation?: boolean;
        credibility_breakdown?: CredibilityBreakdown | null;
        judgment?: RumorJudgment | null;
      };
      if (!response.ok || payload.ok === false) {
        const reasons = Array.isArray(payload.reasons) ? payload.reasons : undefined;
        setModerationError(
          plainModerationError(reasons, "The simulation could not post that rumor. Try a shorter fictional line."),
        );
        return;
      }
      if (payload.read_as) {
        setRumorInterpretation(payload.read_as);
      }
      setRumorBreakdown(payload.credibility_breakdown ?? null);
      setRumorJudgment(payload.judgment ?? null);
      setDraftRumor("");
      if (payload.affects_simulation) {
        playback.resume();
      }
    } catch {
      setModerationError("Can't reach the simulation. It will reconnect when the server is back.");
      setConnection("reconnecting");
    }
  }, [draftRumor, fakeMode, playback, poster, wireFrame.bank_id]);

  const setReducedMotion = useCallback((on: boolean) => {
    reducedMotionRef.current = on;
  }, []);

  return {
    frame: displayFrame,
    mode: fakeMode ? "fake" : "live",
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
    playback: {
      ...playback,
      paused: fakeMode ? !fakePlaying : playback.paused,
      reset: () => {
        playback.reset();
        setDraftRumor("");
        setModerationError(null);
        setRumorInterpretation(null);
        setRumorBreakdown(null);
        setRumorJudgment(null);
        if (fakeMode) {
          setFakePlaying(false);
          fakeStartRef.current = null;
          setFakeRumor("");
          setFakeEffectScale(0);
          setWireFrame(fakeFrame as CrowdFrame);
          setDisplayFrame(fakeFrame as CrowdFrame);
        }
      },
      skipToOutcome: () => {
        if (fakeMode) {
          const elapsed = 98;
          fakeStartRef.current = performance.now() - elapsed * 1000;
          setFakePlaying(true);
          setFakeRumor((current) => current || draftRumor);
        } else {
          playback.skipToOutcome();
        }
      },
      togglePause: () => {
        if (fakeMode) {
          setFakePlaying((playing) => {
            const next = !playing;
            if (!next) {
              fakePausedAtRef.current = performance.now();
            } else if (fakePausedAtRef.current !== null && fakeStartRef.current !== null) {
              fakeStartRef.current += performance.now() - fakePausedAtRef.current;
              fakePausedAtRef.current = null;
            }
            return next;
          });
        } else {
          playback.togglePause();
        }
      },
      changeSpeed: (next: 1 | 2 | 4 | 8) => {
        fakeSpeedRef.current = next;
        playback.changeSpeed(next);
      },
    },
  };
}

export { POSTERS };
