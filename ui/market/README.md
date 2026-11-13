# M30 market screen and crisis room (UI shell)

React + uPlot shell for the market view and crisis room (stream L7). Full chart polish and GPU-backed runs are later waves; this package renders from **fake-engine** snapshots and optional WebSocket frames.

## Layout

- **Market:** uPlot price chart (halt marker overlay), order-book depth bars, halt banner.
- **Crisis room:** accepted/rejected statements with reasons, 16-bar futures fan.

## Fake mode (no GPU, no server)

```bash
cd ui/market
npm install
npm run dev
```

Dev server: **http://127.0.0.1:5174/** (see `vite.config.ts`).

Open `http://127.0.0.1:5174/?fake=1` to load the static demo frame.

### Both monitors (crowd + market)

From the repo root:

```bash
./scripts/run_demo.sh
```

Starts crowd UI on port 5173 and this app on 5174 in fake mode, then opens both in the default browser.

## Live feed (fake engine over WebSocket)

```bash
python3 -m pip install -e "..[server]"   # from repo root: pip install -e ".[server]"
python3 -m ui.market.server              # msgpack frames at registry stream rate (10 Hz)
```

With the Vite dev server proxy, start `npm run dev` and open `http://localhost:5173/` (omit `fake=1`).

Python snapshot API: `ui.market.snapshot.build_market_frame(engine)` → `MarketScreenFrame` (§6.2).
