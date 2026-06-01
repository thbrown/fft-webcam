import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // getUserMedia requires a secure context. localhost is treated as secure,
    // so plain http on localhost is fine for development.
    host: true,
  },
});
