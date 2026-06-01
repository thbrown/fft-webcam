import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Project GitHub Pages serves at https://thbrown.github.io/fft-webcam/, so
  // built asset URLs must be prefixed with the repo name. Dev stays at root.
  base: command === "build" ? "/fft-webcam/" : "/",
  build: {
    // GitHub Pages is configured to serve from the /docs folder on main.
    outDir: "docs",
  },
  server: {
    // getUserMedia requires a secure context. localhost is treated as secure,
    // so plain http on localhost is fine for development.
    host: true,
  },
}));
