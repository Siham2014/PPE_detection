import { defineConfig, mergeConfig } from 'vite';
import baseConfig from './vite.config';

export default mergeConfig(baseConfig, defineConfig({
  plugins: [{
    name: 'ppe-demo-notice',
    transformIndexHtml: {
      order: 'pre',
      handler(html) {
        return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PPE Safety Monitoring — Demo</title></head><body>
<div style="padding:.65rem 1rem;background:#ffedb3;color:#332100;text-align:center;font:600 14px system-ui">Interface demo · Sample indicators and alerts · Live cameras and YOLO inference are unavailable here.</div>
${html}</body></html>`;
      },
    },
  }],
}));
