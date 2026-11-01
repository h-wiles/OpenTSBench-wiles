import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 开发环境把 API 请求代理到 FastAPI (python cli.py serve)
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
