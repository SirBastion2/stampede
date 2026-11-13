# M29 crowd screen (UI shell)

React shell for the depositor point cloud and side panels (cash gauge, time to failure, tipping point, rumor box, poster selector, state ribbon).

## Fake mode (no simulation GPU)

Python builds `CrowdFrame` snapshots from the fake engine (`mocks/l6_crowd_ui.py`) and the contracts in `contracts/schemas.py`:

```bash
pip install -e ".[dev]"
python3 -c "
from engine.registry import build_engine, run_scenario
from tests.test_fakes import panic
from ui.crowd import snapshot_from_engine
import json
from pathlib import Path
engine = build_engine(scenario=panic(1))
engine.run(6)
frame = snapshot_from_engine(engine, t=5.0)
Path('ui/crowd/src/fixtures/fake_frame.json').write_text(json.dumps(frame.model_dump(), indent=2))
"
```

## Frontend

```bash
cd ui/crowd
npm install
npm run dev
```

Dev server: **http://127.0.0.1:5173/** (see `vite.config.ts`).

### Both monitors (crowd + market)

From the repo root:

```bash
./scripts/run_demo.sh
```

Installs npm deps if needed, starts crowd (5173) and market fake mode (5174), then opens both URLs with `xdg-open`.

Defaults to **fake mode** (`VITE_CROWD_FAKE` unset or `1`): loads `src/fixtures/fake_frame.json` and paints with a 2D canvas (Plague Inc.-style cluster spread, no WebGL). Posting a rumor starts a real-time scripted spread (`fakeSpreadPlayback.ts`). Set `VITE_CROWD_DECK=1` to use deck.gl. Set `VITE_CROWD_FAKE=0` and `VITE_CROWD_WS_URL` for live msgpack frames from `ui.server` with client interpolation (`interpolate.ts`).

Playback bar: pause/resume, 1×/2×/4×/8× speed, reset, skip to outcome (`POST /api/control` in live mode).

Wire format: `encode_crowd_frame` / `decode_crowd_frame` in `ui/crowd/frames.py` (msgpack + float32 xy + uint8 state).

Live feed: `VITE_CROWD_FAKE=0` and `ws://127.0.0.1:8000/ws/crowd`; submit rumors via `POST /api/rumor` or the rumor box (`useCrowdFeed`).
