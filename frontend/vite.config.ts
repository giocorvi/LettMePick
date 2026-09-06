import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
    watch: {
      ignored: ["**/node_modules/**", "**/dist/**", "../.venv/**", "../data/**", "../runs/**"],
    },
  },
  test: {
    environment: "jsdom",
    environmentOptions: { jsdom: { url: "http://localhost/" } },
    setupFiles: "./src/test/setup.ts",
    css: true,
    testTimeout: 15_000,
  },
});
