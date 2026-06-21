# 🏃 TrackMotion

**Match the move. Beat your score.**

TrackMotion is a browser-only web app that uses your webcam and a shared browser tab to track two people's skeletal motion in real time and score how closely your movements match a reference video.

No server, no sign-up, no install — just open it in Chrome/Edge and go.

🔗 **Live site:** `https://david-risney.github.io/TrackMotion/`

---

## How it works

1. **Share a tab** — open a dance or exercise video in another tab. Come back to TrackMotion and click *Share Tab*, then pick that tab.
2. **Enable camera** — allow webcam access and stand back so your full body is visible.
3. **Start Tracking** — the AI model loads (~5 MB on first visit), then the live comparison begins.

Both video feeds appear side-by-side with a coloured skeleton overlay.  
A central gauge shows your **match score (0-100%)** updated every frame, and a scrolling history graph lets you track your improvement over time.

---

## How the score is calculated

TrackMotion uses **joint-angle comparison**:

- It measures the angle at eight key joints in both poses: elbows (×2), shoulders (×2), hips (×2), knees (×2).
- Each angle pair is compared; the difference is mapped to a 0-100% similarity (90° difference = 0%, 0° difference = 100%).
- Joints with higher movement impact (knees, elbows) are weighted more heavily.
- The raw score is smoothed frame-to-frame with an exponential moving average to avoid jitter.

---

## Technology

| Concern | Solution |
|---|---|
| Skeleton detection | [TensorFlow.js](https://www.tensorflow.org/js) + [MoveNet SinglePose Lightning](https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4) |
| Reference video capture | `getDisplayMedia` (Screen Capture API) |
| Webcam capture | `getUserMedia` |
| Rendering | HTML5 Canvas |
| Styling | Vanilla CSS (dark theme, CSS custom properties) |
| Framework | **None** — plain HTML, CSS & JS |

---

## Browser requirements

| Feature | Minimum |
|---|---|
| `getDisplayMedia` (tab sharing) | Chrome 72+, Edge 79+, Firefox 66+ |
| `getUserMedia` | All modern browsers |
| WebGL (GPU acceleration) | Chrome 9+, Firefox 4+, Edge 12+ |

> Safari has limited `getDisplayMedia` support; Chrome or Edge recommended.

---

## Running locally

No build step needed — just serve the files:

```bash
npx serve .
# or
python -m http.server 8080
```

Then open `http://localhost:8080`.

---

## Tips for best results

- Make the reference video **full-screen** in the shared tab before sharing.
- Ensure **good lighting** for your webcam — the AI works best with a clear contrast between you and the background.
- Stand far enough back so your **full body** is in frame.
- Wear different colours from your background if possible.