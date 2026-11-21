import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Relative Asset-URLs funktionieren unter / sowie /polytope5d/.
  base: './',
});
