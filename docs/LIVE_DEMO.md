# STAMPEDE live demo

How to run the live crowd and market screens. The project overview is in the top-level `README.md`; the design is in `docs/ARCHITECTURE.md`.

## Live demo (crowd + market)

Offline CPU defaults (`STAMPEDE_OFFLINE=1`; `run_live.sh` runs the calibrated real modules, `STAMPEDE_MODE=fake` plays the scripted arc):

```bash
pip install -e '.[server,dev]'
./scripts/run_live.sh
```

Frontend build order: `ui/crowd` compiles `../market/src`, so install `ui/market` first, or `ui/crowd` fails with TS2307/TS2875 for `react`, `react/jsx-runtime` and `@msgpack/msgpack`:

```bash
make ui-build   # = (cd ui/market && npm ci) && (cd ui/crowd && npm ci && npm run build) && (cd ui/market && npm run build)
```

- API / WebSockets: `http://127.0.0.1:8000` (`/ws/crowd`, `/ws/market`, `POST /api/rumor`, `POST /api/control`)
- Playback controls (crowd header): **Pause / Resume**, **1×–8× speed**, **Reset**, **Skip to outcome**. Pacing reads `M29 UI live crisis sim minutes`, `M29 UI live playback target seconds`, and `M03 sim tick minutes` from `params/registry.csv` (default ~120 sim minutes in ~105 real seconds at 1×). Frames stream at `M29 UI client stream fps` (30 Hz); slow clients drop frames, not engine ticks.
- Demo page (crowd and market together): `http://127.0.0.1:5173` (set `VITE_CROWD_FAKE=0` in `run_live.sh` for live frames)
- Market UI alone: `http://127.0.0.1:5174` (omit `?fake=1` for live)

## Two-minute walkthrough

Kestrel Bank is fictional. This is a rehearsal tool for bank risk teams and supervisors.

1. Open the demo page. The hint says to type a rumor and press Enter.
2. Post `I heard Kestrel Bank lost billions on bonds` as the VC poster.
3. Gray squares spread through the clusters and KSTL falls. Pause, change speed (1×–8×), or Skip to outcome.
4. The crisis room lists accepted and rejected statements. A real name such as Chase is blocked, with a message to use a fictional bank.
5. Reset is one click and does not need a reload.

What the page claims, and nothing more: this offline run draws the Tier-0 population (100,000 depositors) as 2,048 orbs on the CPU fake engine. The registry design-target population is 1,000,000. Calibration against 2023 runs is not fitted here. Context: in March 2023, a rumor emptied a real bank in about 36 hours. This demo rehearses that on a fictional bank only.

## Future plans

- Stretch the operating-cash and withdrawal curves so a 1× run shows a gradual crowd change across the ~2 minute window.
- Show measured tick time from the metrics stream. CPU today; GPU only when that backend is actually running. Do not print a sub-5 ms figure until it is measured.
- Replace crisis-room statement ids with the plain-language lines the engine already scores.

## Credits

Python: FastAPI, Uvicorn, msgpack, httpx, pytest. UI: React, Vite, TypeScript. Versions and the rest of the tree are in `pyproject.toml` and `ui/crowd/package.json` / `ui/market/package.json`.

Environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STAMPEDE_OFFLINE` | `1` | No paid LLM calls |
| `STAMPEDE_MODE` | `fake` (`scripts/run_live.sh`: `real`) | `real` runs the calibrated engine (`M29 live engine modules`) and the live crowd and price follow it; `fake` plays the scripted arc |
| `STAMPEDE_BACKEND` | `cpu` | `gpu` uses CuPy when installed |
| `STAMPEDE_PORT` | `8000` | Live server port |

## Tests (offline)

```bash
STAMPEDE_OFFLINE=1 python3 -m pytest tests/test_live_rumor_e2e.py tests/test_live_pacing.py -q
```
