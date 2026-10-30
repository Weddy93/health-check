import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// In development the Flask API runs on http://localhost:5000. Proxy /api
// requests so relative URLs in src/api.js work identically in dev and in
// production (where Flask serves the SPA itself).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
