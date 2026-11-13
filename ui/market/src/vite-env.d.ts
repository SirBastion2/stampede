/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MARKET_WS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
