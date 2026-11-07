import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Development config (the production build uses vite.static.config.ts).
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
});
