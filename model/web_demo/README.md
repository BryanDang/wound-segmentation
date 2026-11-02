# Web demo: wound segmentation

Browser-based inference demo for the wound segmentation model, deployed with TensorFlow.js.

Status: only this README is present in `web_demo/`. The files, scripts and API docs it described are not in the repository. Most of the original text is damaged; only the readable parts are kept below.

## Features

- Real-time inference: client-side wound segmentation with optimized TinyML models.
- Interface: upload an image and visualize results.
- Performance metrics: real-time speed indicators.
- Educational content: model explanations.
- Responsive: desktop, tablet and mobile.

## Technology stack

- Frontend framework: text damaged in source.
- ML runtime: TensorFlow.js.
- UI library: text damaged in source.
- Visualization: text damaged in source.
- Model format: TensorFlow.js (`model.json` plus weight shard files).
- Model size: under 5 MB.
- Precision: INT8 quantization for the speed/size balance.
- Backend: WebGL.

## Setup

1. Change into `web_demo/`.
2. Install dependencies (build tools, if any): text damaged in source.
3. Serve locally with a static HTTP server on port 8080 (`npx http-server -p 8080 -c-1`, partly damaged in source).
4. Open `http://localhost:8080`.

## Model deployment

1. Convert the PyTorch model to TensorFlow.js with `src.deployment` (command damaged in source), writing to `web_demo/model/`. Source model path: `models/optimized/best_model`.
2. Verify the model files in `web_demo/model/`: `model.json`, weight shard files, metadata.
3. Test loading in the browser:
```javascript
const model = await tf.loadLayersModel('./model/model.json');
console.log('Model loaded successfully:', model);
```

## User workflow

1. Load the application. The model loads in the browser.
2. Upload an image or use the provided examples.
3. Run inference: click "Analyze Wound".
4. View results: overlay and metrics.
5. Download the processed results.

## Advanced features

- Batch processing: upload multiple images for comparison.
- Threshold adjustment: fine-tune sensitivity.
- Overlay customization: adjust color and transparency.
- Performance monitoring: view speed and memory usage.
- Export options: download results.

## API reference

Model interface (class name partly damaged in source, begins `WoundSegmentationMod`):
- `loadModel(modelUrl)`: load the TensorFlow.js model.
- `predict(image)`: returns `{ segmentation: tensor, confidence: ... }` (rest damaged in source).
- `preprocessImage(image, targetSize = [640, 640])`
- `postprocessPrediction(prediction, originalSize)`

Image processing (class name damaged in source):
- `resizeImage(image, targetWidth, targetHeight)`
- `createOverlay(originalImage, segmentationMask, color, ...)`
- `calculateMetrics(segmentationMask)`: wound area, perimeter, etc.

## Performance optimization

Model optimization:
- Quantization: INT8, reduces model size by about 75%.
- Pruning: removes redundant parameters (rest damaged in source).
- Graph optimization: TensorFlow.js optimizations.
- WebGL backend: GPU acceleration.

Runtime optimization:
- Lazy loading: load the model when needed.
- Memory management: avoid leaks.
- Caching: preprocessed images and model outputs.
- Progressive enhancement: fallback for older browsers (partly damaged in source).

## Browser compatibility

Supported browsers:
- Chrome 70+ (recommended)
- Firefox 65+
- Safari 12+
- Edge 79+

Mobile support:
- Safari 12+
- Chrome Mobile 70+

Internet Explorer entry: text damaged in source.

Feature detection checks for TensorFlow.js and WebGL availability. Code damaged in source.

## Clinical considerations

- Disclaimer: states AI limitations.
- Medical advice: not a medical assessment.
- Accuracy: displays confidence and estimates.
- Privacy: processing is performed in the browser (partly damaged in source).

## Educational content

- How it works: explanation of the AI model and methodology.
- Limitations: model limitations.
- Clinical context: wound assessment practices.
- References: links to validation studies.

## Hosting options

- GitHub Pages: demonstration hosting.
- Netlify: advanced deployment.
- Vercel: optimized distribution.
- AWS S3 with CloudFront (partly damaged in source).

CDN integration:
```html
<!-- TensorFlow.js CDN -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>

<!-- Model loading -->
<script>
const modelUrl = 'https://your-cdn.com/wound-seg/model.json';
</script>
```

## Progressive web app

- Service worker: offline functionality.
- App manifest: installable app.
- Caching: model and static assets.

## Testing

Text damaged in source. Two test commands were listed: model integration tests and performance benchmark tests (command names damaged).

## Maintenance

Model updates:
1. Convert the new model with the deployment script.
2. Update `web_demo/model/`.
3. Test accuracy and performance.
4. Update metadata.

Security updates:
- Dependency updates.
- Header configuration.
- Content Security Policy (CSP).
- HTTPS.

Performance monitoring:
- Track latency.
- Memory usage.
- Analyze usage patterns.
- A/B test UI improvements.
