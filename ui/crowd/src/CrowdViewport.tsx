import { useEffect, useMemo, useRef } from "react";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { clusterCenters, clusterStats, communityIds, DEFAULT_COMMUNITY_COUNT } from "./clusterLayout";
import { MAX_DRAWN_ORBS, ORBS_PER_TICK, orbRadius, stateRgb } from "./orbStyle";
import type { CrowdFrame } from "./types";

function mix(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  return [
    Math.round(a[0] + (b[0] - a[0]) * clamped),
    Math.round(a[1] + (b[1] - a[1]) * clamped),
    Math.round(a[2] + (b[2] - a[2]) * clamped),
  ];
}

function drawnCount(frame: CrowdFrame): number {
  const xy = Math.floor(frame.positions_xy.length / 2);
  return Math.max(0, Math.min(frame.point_count, frame.state.length, xy, MAX_DRAWN_ORBS));
}

interface CrowdViewportProps {
  frame: CrowdFrame;
  reducedMotion?: boolean;
}

export function CrowdViewport({ frame, reducedMotion = false }: CrowdViewportProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef(0);
  const shownRef = useRef<number[] | null>(null);
  const colorRef = useRef<Float32Array | null>(null);
  const targetRef = useRef(frame);
  const reducedRef = useRef(reducedMotion);
  targetRef.current = frame;
  reducedRef.current = reducedMotion;
  const useDeck = import.meta.env.VITE_CROWD_DECK === "1";
  const useCanvasFallback = !useDeck;
  const limit = drawnCount(frame);

  const layoutMeta = useMemo(() => {
    const ids = communityIds(limit, DEFAULT_COMMUNITY_COUNT);
    const centers = clusterCenters(frame.positions_xy, ids, DEFAULT_COMMUNITY_COUNT);
    const stats = clusterStats(frame.state.slice(0, limit), ids, DEFAULT_COMMUNITY_COUNT);
    return { ids, centers, stats };
  }, [frame, limit]);

  const layerData = useMemo(() => {
    const data = [];
    for (let i = 0; i < limit; i += 1) {
      const state = frame.state[i] ?? 0;
      const rgb = stateRgb(state);
      data.push({
        position: [frame.positions_xy[i * 2], frame.positions_xy[i * 2 + 1]] as [number, number],
        radius: orbRadius(frame.uninsured_mask[i] ?? false, frame.balance_norm[i] ?? 0.5),
        color: [rgb[0], rgb[1], rgb[2], 220] as [number, number, number, number],
      });
    }
    return data;
  }, [frame, limit]);

  useEffect(() => {
    if (!useCanvasFallback || !canvasRef.current) {
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    if (!shownRef.current || shownRef.current.length !== limit) {
      const seed = targetRef.current.state.slice(0, limit);
      const colors = new Float32Array(limit * 3);
      for (let i = 0; i < limit; i += 1) {
        const rgb = stateRgb(seed[i] ?? 0);
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
      shownRef.current = seed;
      colorRef.current = colors;
    }

    const { ids, centers, stats } = layoutMeta;

    const paint = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (width < 2 || height < 2) {
        rafRef.current = requestAnimationFrame(paint);
        return;
      }
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const pixelW = Math.floor(width * dpr);
      const pixelH = Math.floor(height * dpr);
      if (canvas.width !== pixelW || canvas.height !== pixelH) {
        canvas.width = pixelW;
        canvas.height = pixelH;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const shown = shownRef.current;
      const colors = colorRef.current;
      const target = targetRef.current.state;
      if (!shown || !colors) {
        rafRef.current = requestAnimationFrame(paint);
        return;
      }
      const pending: number[] = [];
      for (let i = 0; i < limit; i += 1) {
        if ((shown[i] ?? 0) !== (target[i] ?? 0)) pending.push(i);
      }
      const snap = reducedRef.current;
      const budget = snap
        ? pending.length
        : Math.min(pending.length, Math.max(ORBS_PER_TICK, Math.ceil(pending.length / 20)));
      for (let n = 0; n < budget; n += 1) {
        const index = pending[n];
        shown[index] = target[index] ?? 0;
      }
      const ease = snap ? 1 : 0.22;
      for (let i = 0; i < limit; i += 1) {
        const rgb = stateRgb(shown[i] ?? 0);
        colors[i * 3] += (rgb[0] - colors[i * 3]) * ease;
        colors[i * 3 + 1] += (rgb[1] - colors[i * 3 + 1]) * ease;
        colors[i * 3 + 2] += (rgb[2] - colors[i * 3 + 2]) * ease;
      }

      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (let i = 0; i < limit; i += 1) {
        const x = frame.positions_xy[i * 2];
        const y = frame.positions_xy[i * 2 + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          continue;
        }
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      if (!Number.isFinite(minX)) {
        rafRef.current = requestAnimationFrame(paint);
        return;
      }
      const worldW = Math.max(0.001, maxX - minX);
      const worldH = Math.max(0.001, maxY - minY);
      const scale = Math.min((width * 0.9) / worldW, (height * 0.86) / worldH);
      const offsetX = width / 2 - ((minX + maxX) / 2) * scale;
      const offsetY = height / 2 - ((minY + maxY) / 2) * scale;
      const project = (x: number, y: number) => [offsetX + x * scale, offsetY + y * scale] as const;

      for (const center of centers) {
        const stat = stats[center.id];
        if (!stat || stat.total === 0) {
          continue;
        }
        let maxReach = 0;
        for (let i = 0; i < limit; i += 1) {
          if ((ids[i] ?? 0) !== center.id) {
            continue;
          }
          const dx = frame.positions_xy[i * 2] - center.x;
          const dy = frame.positions_xy[i * 2 + 1] - center.y;
          const reach = Math.hypot(dx, dy);
          if (reach > maxReach) {
            maxReach = reach;
          }
        }
        let nearest = Infinity;
        for (const other of centers) {
          if (other.id === center.id || (stats[other.id]?.total ?? 0) === 0) {
            continue;
          }
          nearest = Math.min(nearest, Math.hypot(other.x - center.x, other.y - center.y));
        }
        const reach = Math.min(maxReach * 1.12, Number.isFinite(nearest) ? nearest * 0.46 : maxReach);
        if (reach <= 0) {
          continue;
        }
        const redShare = stat.withdrew / stat.total;
        const tint = mix([22, 64, 48], [120, 28, 36], Math.min(1, redShare * 1.2));
        const [cx, cy] = project(center.x, center.y);
        const radius = reach * scale;
        const gradient = ctx.createRadialGradient(cx, cy, radius * 0.15, cx, cy, radius);
        const alpha = 0.1 + redShare * 0.42;
        gradient.addColorStop(0, `rgba(${tint[0]}, ${tint[1]}, ${tint[2]}, ${alpha})`);
        gradient.addColorStop(1, `rgba(${tint[0]}, ${tint[1]}, ${tint[2]}, 0)`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      const drawOrb = (i: number) => {
        const x = frame.positions_xy[i * 2];
        const y = frame.positions_xy[i * 2 + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          return;
        }
        const [px, py] = project(x, y);
        const uninsured = frame.uninsured_mask[i] ?? false;
        const radius = orbRadius(uninsured, frame.balance_norm[i] ?? 0.5);
        const r = Math.round(colors[i * 3]);
        const g = Math.round(colors[i * 3 + 1]);
        const b = Math.round(colors[i * 3 + 2]);
        const state = shown[i] ?? 0;
        ctx.fillStyle = `rgba(${r},${g},${b},0.92)`;
        ctx.beginPath();
        if (state === 2) {
          ctx.moveTo(px, py - radius);
          ctx.lineTo(px + radius, py + radius * 0.85);
          ctx.lineTo(px - radius, py + radius * 0.85);
          ctx.closePath();
        } else if (state === 1) {
          ctx.rect(px - radius, py - radius, radius * 2, radius * 2);
        } else {
          ctx.arc(px, py, radius, 0, Math.PI * 2);
        }
        ctx.fill();
        if (uninsured) {
          ctx.strokeStyle = "rgba(255,255,255,0.85)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      };
      for (let i = 0; i < limit; i += 1) {
        if (frame.uninsured_mask[i]) continue;
        drawOrb(i);
      }
      for (let i = 0; i < limit; i += 1) {
        if (!frame.uninsured_mask[i]) continue;
        drawOrb(i);
      }

      rafRef.current = requestAnimationFrame(paint);
    };

    rafRef.current = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(rafRef.current);
  }, [layoutMeta, limit, useCanvasFallback]);

  if (useCanvasFallback) {
    return (
      <canvas
        ref={canvasRef}
        className="crowd-canvas"
        aria-label="Crowd neighborhoods"
      />
    );
  }

  const layers = [
    new ScatterplotLayer({
      id: "crowd-orbs",
      data: layerData,
      getPosition: (d) => d.position,
      getRadius: (d) => d.radius,
      getFillColor: (d) => d.color,
      radiusUnits: "pixels",
      pickable: false,
    }),
  ];

  return (
    <DeckGL
      initialViewState={{ longitude: 0, latitude: 0, zoom: 1, pitch: 0, bearing: 0 }}
      controller
      viewState={{ longitude: 0, latitude: 0, zoom: 1 }}
      layers={layers}
    />
  );
}
