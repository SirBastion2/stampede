import { decode } from "@msgpack/msgpack";
import { useEffect, useRef, useState } from "react";
import { FAKE_FRAME } from "../fakeFrame";
import { fakeMarketFrame } from "../fakeSpreadMarket";
import type { MarketScreenFrame } from "../types";

const WS_URL =
  (import.meta.env.VITE_MARKET_WS as string | undefined) ?? "ws://127.0.0.1:8000/ws/market";

function useFakeMode(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  const params = new URLSearchParams(window.location.search);
  return params.get("fake") === "1" || params.get("offline") === "1";
}

export function useMarketFeed(): {
  frame: MarketScreenFrame | null;
  mode: "fake" | "live" | "connecting";
  error: string | null;
} {
  const fake = useFakeMode();
  const [wireFrame, setWireFrame] = useState<MarketScreenFrame | null>(fake ? FAKE_FRAME : null);
  const [frame, setFrame] = useState<MarketScreenFrame | null>(fake ? FAKE_FRAME : null);
  const [mode, setMode] = useState<"fake" | "live" | "connecting">(fake ? "fake" : "connecting");
  const [error, setError] = useState<string | null>(null);
  const prevRef = useRef<MarketScreenFrame | null>(null);
  const blendStartRef = useRef(0);

  useEffect(() => {
    if (!fake) {
      return;
    }
    const start = performance.now();
    let raf = 0;
    const loop = () => {
      const elapsed = (performance.now() - start) / 1000;
      setFrame(fakeMarketFrame(elapsed));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [fake]);

  useEffect(() => {
    if (fake) {
      return;
    }
    let stopped = false;
    let socket: WebSocket | null = null;
    let retry = 0;
    let timer = 0;

    const connect = () => {
      if (stopped) return;
      if (retry > 0) {
        setMode("connecting");
        setError("Reconnecting to the simulation…");
      }
      try {
        socket = new WebSocket(WS_URL);
      } catch {
        timer = window.setTimeout(connect, Math.min(4000, 400 * (retry + 1)));
        retry += 1;
        return;
      }
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        retry = 0;
        setMode("live");
        setError(null);
      };
      socket.onmessage = (event) => {
        if (typeof event.data === "string") return;
        const data = decode(new Uint8Array(event.data as ArrayBuffer)) as MarketScreenFrame;
        setWireFrame((prev) => {
          prevRef.current = prev;
          blendStartRef.current = performance.now();
          return data;
        });
        setMode("live");
        setError(null);
      };
      socket.onclose = () => {
        if (stopped) return;
        setMode("connecting");
        setError("Reconnecting to the simulation…");
        timer = window.setTimeout(connect, Math.min(4000, 400 * (retry + 1)));
        retry += 1;
      };
    };

    setMode("connecting");
    setError("Connecting to the market feed…");
    connect();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      socket?.close();
    };
  }, [fake]);

  useEffect(() => {
    if (fake) {
      return;
    }
    let raf = 0;
    const tick = () => {
      const next = wireFrame;
      const prev = prevRef.current;
      if (next && prev && prev !== next) {
        const reduce =
          window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
          document.documentElement.dataset.motion === "reduce";
        const alpha = reduce
          ? 1
          : Math.min(1, (performance.now() - blendStartRef.current) / (1000 / 30));
        const blendPrice = (a: number, b: number) => a + (b - a) * alpha;
        const prices = next.prices.map((row, i) => {
          const prevRow = prev.prices[i] ?? prev.prices[prev.prices.length - 1];
          return [row[0], blendPrice(prevRow[1], row[1])] as [number, number];
        });
        setFrame({ ...next, prices, t: blendPrice(prev.t, next.t) });
      } else if (next) {
        setFrame(next);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [fake, wireFrame]);

  return { frame, mode, error };
}
