/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CROWD_FAKE?: string;
  readonly VITE_CROWD_DECK?: string;
  readonly VITE_CROWD_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
