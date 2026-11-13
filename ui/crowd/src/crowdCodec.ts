import { decode } from "@msgpack/msgpack";
import type { CrowdFrame, CrowdPanels, StateRibbon } from "./types";

/** Reinterpret a msgpack bin as float32. A Uint8Array must not be passed to
 * `new Float32Array(typedArray)` — that copies byte values, not the floats,
 * and piles every orb on one huge circle. */
export function readFloat32Blob(value: unknown): number[] {
  if (value instanceof ArrayBuffer) {
    return Array.from(new Float32Array(value.slice(0)));
  }
  if (ArrayBuffer.isView(value)) {
    const view = value as ArrayBufferView;
    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    const aligned = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(aligned).set(bytes);
    const count = Math.floor(aligned.byteLength / 4);
    return Array.from(new Float32Array(aligned, 0, count));
  }
  if (Array.isArray(value)) {
    return value.map((item) => Number(item));
  }
  return [];
}

export function decodeCrowdFrame(buffer: ArrayBuffer): CrowdFrame {
  const raw = decode(new Uint8Array(buffer)) as Record<string, unknown>;
  const pointCount = Number(raw.point_count);
  const positions_xy = readFloat32Blob(raw.xy);
  const balance_norm = readFloat32Blob(raw.balance);
  const stateBlob = raw.state as Uint8Array;
  const uninsuredBlob = raw.uninsured as Uint8Array;
  const panels = raw.panels as CrowdPanels;
  const state = Array.from(stateBlob.slice(0, pointCount));
  const uninsured_mask = Array.from(uninsuredBlob.slice(0, pointCount)).map((byte) => byte !== 0);

  return {
    t: Number(raw.t),
    seed: Number(raw.seed),
    bank_id: (raw.bank_id as string | null | undefined) ?? null,
    point_count: pointCount,
    positions_xy,
    state,
    balance_norm,
    uninsured_mask,
    panels: {
      ...panels,
      state_ribbon: panels.state_ribbon as StateRibbon,
    },
  };
}
