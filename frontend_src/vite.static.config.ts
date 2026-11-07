import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));

// A plain Vite client build shares the same generated TanStack route tree as
// the managed preview, but contains no Start server or server functions.
export default defineConfig({
  root: `${root}static`,
  cacheDir: `${root}node_modules/.vite-static`,
  base: "./",
  envDir: root,
  publicDir: `${root}public`,
  define: { "import.meta.env.VITE_STATIC_SPA": JSON.stringify("true") },
  plugins: [
    tanstackRouter({ target: "react", routesDirectory: `${root}src/routes`, generatedRouteTree: `${root}src/routeTree.gen.ts`, autoCodeSplitting: false }),
    react(),
    tailwindcss(),
  ],
  resolve: { alias: { "@": `${root}src` } },
  build: { outDir: `${root}dist`, emptyOutDir: true },
});