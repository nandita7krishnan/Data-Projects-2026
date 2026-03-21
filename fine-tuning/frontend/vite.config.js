import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/pitch': 'http://localhost:8000',
      '/characters': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
