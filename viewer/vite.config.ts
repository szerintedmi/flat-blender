import { defineConfig } from "vite";

export default defineConfig({
  base: "./", // dist/ works from any path (file server, GitHub Pages, ...)
  build: { target: "es2022", chunkSizeWarningLimit: 4000 }, // Spark bundles its WASM: one big chunk
});
