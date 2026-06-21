/* ══════════════════════════════════════════════════════════════
   TrackMotion — app.js
   Client-side only. No bundler. Vanilla JS + TensorFlow.js.
   ══════════════════════════════════════════════════════════════ */
'use strict';

/* ── Skeleton connections (MoveNet keypoint indices) ───────── */
const CONNECTIONS = [
  [0, 1], [0, 2],           // nose → eyes
  [1, 3], [2, 4],           // eyes → ears
  [5, 6],                   // left shoulder → right shoulder
  [5, 7], [7, 9],           // left arm
  [6, 8], [8, 10],          // right arm
  [5, 11], [6, 12],         // shoulder → hip
  [11, 12],                 // left hip → right hip
  [11, 13], [13, 15],       // left leg
  [12, 14], [14, 16],       // right leg
];

/*
 * Joint angle triplets: [indexA, indexVertex, indexB, weight]
 * Measures the angle at `vertex` formed by points A and B.
 * Higher weight = contributes more to the match score.
 */
const ANGLE_TRIPLETS = [
  [5,  7,  9,  2.0],   // left  elbow   (shoulder→elbow←wrist)
  [6,  8, 10,  2.0],   // right elbow
  [11, 5,  7,  1.5],   // left  shoulder (hip→shoulder←elbow)
  [12, 6,  8,  1.5],   // right shoulder
  [5,  11, 13, 1.2],   // left  hip     (shoulder→hip←knee)
  [6,  12, 14, 1.2],   // right hip
  [11, 13, 15, 1.8],   // left  knee    (hip→knee←ankle)
  [12, 14, 16, 1.8],   // right knee
];

/* Minimum keypoint confidence to use a point */
const KP_MIN_SCORE = 0.25;

/* Score smoothing (exponential moving average) */
const EMA_ALPHA = 0.12;

/* Angle difference that maps to 0% similarity (degrees) */
const MAX_ANGLE_DIFF = 90;

/* Circumference of the SVG gauge ring (r = 60 → 2π×60 ≈ 376.99) */
const RING_CIRCUM = 2 * Math.PI * 60;

/* Number of historical score samples to draw */
const HISTORY_LEN = 180;

/* Skeleton colours */
const REF_COLOR = '#4ecdc4';   // teal  — reference
const CAM_COLOR = '#ff6b6b';   // coral — user

/* ─────────────────────────────────────────────────────────── */

/* ── Application state ─────────────────────────────────────── */
const state = {
  refStream:      null,   // MediaStream from getDisplayMedia
  camStream:      null,   // MediaStream from getUserMedia
  detector:       null,   // poseDetection.PoseDetector
  running:        false,
  rafId:          null,
  smoothedScore:  null,   // EMA-smoothed score (0-100 or null)
  history:        [],     // circular buffer of smoothed scores
};

/* ── DOM references ────────────────────────────────────────── */
function $(id) { return document.getElementById(id); }

const dom = {
  setupPanel:      $('setup-panel'),
  loadingOverlay:  $('loading-overlay'),
  trackingPanel:   $('tracking-panel'),

  btnShare:        $('btn-share'),
  btnCamera:       $('btn-camera'),
  btnStart:        $('btn-start'),
  btnStop:         $('btn-stop'),

  statusMsg:       $('status-msg'),
  checkShare:      $('check-share'),
  checkCamera:     $('check-camera'),

  stepShare:       $('step-share'),
  stepCamera:      $('step-camera'),

  refVideo:        $('ref-video'),
  refCanvas:       $('ref-canvas'),
  refBadge:        $('ref-badge'),

  camVideo:        $('cam-video'),
  camCanvas:       $('cam-canvas'),
  camBadge:        $('cam-badge'),

  gaugeFill:       $('gauge-fill'),
  scoreValue:      $('score-value'),
  feedbackText:    $('feedback-text'),
  historyCanvas:   $('history-canvas'),
};

/* ─────────────────────────────────────────────────────────── */

/* ── Utilities ─────────────────────────────────────────────── */

function setStatus(msg, isError = false) {
  dom.statusMsg.textContent = msg;
  dom.statusMsg.className = 'status-msg' + (isError ? ' error' : '');
}

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

/** Resolve once a video element has valid video data. */
function videoReady(video) {
  return new Promise(resolve => {
    if (video.readyState >= 2) { resolve(); return; }
    video.addEventListener('loadeddata', resolve, { once: true });
  });
}

/* ─────────────────────────────────────────────────────────── */

/* ── Pose math ─────────────────────────────────────────────── */

/**
 * Compute the angle (degrees) at `vertex` formed by the rays
 * vertex→a and vertex→b.
 */
function angleAt(a, vertex, b) {
  const ax = a.x - vertex.x, ay = a.y - vertex.y;
  const bx = b.x - vertex.x, by = b.y - vertex.y;
  const dot   = ax * bx + ay * by;
  const cross = Math.abs(ax * by - ay * bx);
  return Math.atan2(cross, dot) * (180 / Math.PI);
}

/**
 * Extract joint angles from a keypoints array.
 * Returns an array of { key, angle, weight } for visible joints.
 */
function extractAngles(kps) {
  const out = [];
  for (const [ai, vi, bi, weight] of ANGLE_TRIPLETS) {
    const pa = kps[ai], pv = kps[vi], pb = kps[bi];
    if (
      pa && pv && pb &&
      pa.score >= KP_MIN_SCORE &&
      pv.score >= KP_MIN_SCORE &&
      pb.score >= KP_MIN_SCORE
    ) {
      out.push({
        key: `${ai}-${vi}-${bi}`,
        angle: angleAt(pa, pv, pb),
        weight,
      });
    }
  }
  return out;
}

/**
 * Compare two keypoint arrays and return a score (0-100).
 * Returns null when no shared visible joints are found.
 */
function comparePoses(refKps, camKps) {
  const refAngles = extractAngles(refKps);
  const camAngles = extractAngles(camKps);
  if (refAngles.length === 0 || camAngles.length === 0) return null;

  const camMap = new Map(camAngles.map(a => [a.key, a]));

  let totalWeight = 0;
  let weightedSum = 0;

  for (const ref of refAngles) {
    const cam = camMap.get(ref.key);
    if (!cam) continue;
    const diff       = Math.abs(ref.angle - cam.angle);
    const similarity = Math.max(0, 1 - diff / MAX_ANGLE_DIFF);
    weightedSum  += similarity * ref.weight;
    totalWeight  += ref.weight;
  }

  if (totalWeight === 0) return null;
  return Math.round((weightedSum / totalWeight) * 100);
}

/* ─────────────────────────────────────────────────────────── */

/* ── Skeleton rendering ────────────────────────────────────── */

/**
 * Resize canvas logical pixels to match the video's natural resolution.
 * CSS handles scaling to the display size.
 */
function syncCanvasSize(canvas, video) {
  const w = video.videoWidth  || 640;
  const h = video.videoHeight || 360;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width  = w;
    canvas.height = h;
  }
}

/**
 * Clear and re-draw skeleton on `canvas` using MoveNet `keypoints`.
 * Coordinates from TF.js are already in video-pixel space.
 */
function drawSkeleton(canvas, keypoints, color) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  /* --- connections ----------------------------------------- */
  ctx.strokeStyle = color;
  ctx.lineWidth   = 3;
  ctx.lineCap     = 'round';

  for (const [ai, bi] of CONNECTIONS) {
    const a = keypoints[ai], b = keypoints[bi];
    if (!a || !b || a.score < KP_MIN_SCORE || b.score < KP_MIN_SCORE) continue;
    ctx.globalAlpha = Math.min(a.score, b.score);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  /* --- keypoints ------------------------------------------- */
  ctx.globalAlpha = 1;
  for (const kp of keypoints) {
    if (kp.score < KP_MIN_SCORE) continue;
    /* coloured ring */
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
    ctx.fill();
    /* white centre dot */
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function clearCanvas(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

/* ─────────────────────────────────────────────────────────── */

/* ── Score display ─────────────────────────────────────────── */

function scoreColor(score) {
  if (score >= 90) return 'var(--green)';
  if (score >= 70) return 'var(--yellow)';
  if (score >= 50) return 'var(--orange)';
  return 'var(--red)';
}

function scoreFeedback(score) {
  if (score === null) return 'Waiting for poses…';
  if (score >= 90) return '🔥 Excellent match!';
  if (score >= 75) return '👍 Great job!';
  if (score >= 60) return '🙂 Good, keep going!';
  if (score >= 40) return '💪 Keep trying!';
  return '👀 Watch the reference closely';
}

function updateScoreDisplay(score) {
  if (score === null) {
    dom.scoreValue.textContent = '--';
    dom.scoreValue.style.color = '';
    dom.gaugeFill.style.strokeDashoffset = RING_CIRCUM;
    dom.gaugeFill.style.stroke = 'var(--accent)';
    dom.feedbackText.textContent = scoreFeedback(null);
    dom.feedbackText.style.color = '';
    return;
  }

  const clr = scoreColor(score);
  dom.scoreValue.textContent    = score;
  dom.scoreValue.style.color    = clr;
  dom.gaugeFill.style.stroke    = clr;
  dom.gaugeFill.style.strokeDashoffset =
    RING_CIRCUM * (1 - Math.min(score, 100) / 100);
  dom.feedbackText.textContent  = scoreFeedback(score);
  dom.feedbackText.style.color  = clr;
}

/* ─────────────────────────────────────────────────────────── */

/* ── History graph ─────────────────────────────────────────── */

function drawHistoryGraph() {
  const canvas = dom.historyCanvas;
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  /* horizontal grid lines at 25 / 50 / 75 / 100 % */
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth   = 1;
  [25, 50, 75, 100].forEach(pct => {
    const y = H - (pct / 100) * H;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  });

  const hist = state.history;
  if (hist.length < 2) return;

  /* gradient fill under the line */
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0,   'rgba(78,205,196,.35)');
  grad.addColorStop(1,   'rgba(78,205,196,.02)');

  const toY = v => H - (Math.max(0, Math.min(100, v)) / 100) * H;
  const len = Math.min(hist.length, HISTORY_LEN);
  const slice = hist.slice(-len);
  const step  = W / Math.max(len - 1, 1);

  /* filled area */
  ctx.beginPath();
  slice.forEach((v, i) => {
    const x = i * step, y = (v === null) ? H : toY(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo((slice.length - 1) * step, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  /* line */
  ctx.beginPath();
  ctx.strokeStyle = '#4ecdc4';  /* --accent hex value; canvas doesn't resolve CSS vars */
  ctx.lineWidth   = 2;
  ctx.lineJoin    = 'round';
  let drawing = false;
  slice.forEach((v, i) => {
    if (v === null) { drawing = false; return; }
    const x = i * step, y = toY(v);
    if (!drawing) { ctx.moveTo(x, y); drawing = true; }
    else            ctx.lineTo(x, y);
  });
  ctx.stroke();
}

/* ─────────────────────────────────────────────────────────── */

/* ── Main processing loop ──────────────────────────────────── */

async function processFrame() {
  if (!state.running) return;

  const { refVideo, refCanvas, camVideo, camCanvas } = dom;

  /* skip until both videos have current data (readyState ≥ HAVE_CURRENT_DATA) */
  if (refVideo.readyState < 2 || camVideo.readyState < 2) {
    state.rafId = requestAnimationFrame(processFrame);
    return;
  }

  /* sync canvas resolution to video */
  syncCanvasSize(refCanvas, refVideo);
  syncCanvasSize(camCanvas, camVideo);

  /* detect poses in parallel */
  let refPoses = null, camPoses = null;
  try {
    [refPoses, camPoses] = await Promise.all([
      state.detector.estimatePoses(refVideo),
      state.detector.estimatePoses(camVideo),
    ]);
  } catch (err) {
    console.warn('[TrackMotion] pose estimation error:', err);
  }

  const refPose = refPoses?.[0] ?? null;
  const camPose = camPoses?.[0] ?? null;

  /* draw skeletons */
  if (refPose) {
    drawSkeleton(refCanvas, refPose.keypoints, REF_COLOR);
    dom.refBadge.textContent = 'Pose detected';
    dom.refBadge.className   = 'pose-badge detected';
  } else {
    clearCanvas(refCanvas);
    dom.refBadge.textContent = 'No pose detected';
    dom.refBadge.className   = 'pose-badge';
  }

  if (camPose) {
    drawSkeleton(camCanvas, camPose.keypoints, CAM_COLOR);
    dom.camBadge.textContent = 'Pose detected';
    dom.camBadge.className   = 'pose-badge detected';
  } else {
    clearCanvas(camCanvas);
    dom.camBadge.textContent = 'No pose detected';
    dom.camBadge.className   = 'pose-badge';
  }

  /* compare and smooth score */
  const rawScore = (refPose && camPose)
    ? comparePoses(refPose.keypoints, camPose.keypoints)
    : null;

  if (rawScore !== null) {
    state.smoothedScore = (state.smoothedScore === null)
      ? rawScore
      : Math.round(EMA_ALPHA * rawScore + (1 - EMA_ALPHA) * state.smoothedScore);
  } else if (state.smoothedScore !== null) {
    /* slowly decay score when pose is lost */
    state.smoothedScore = Math.max(0, state.smoothedScore - 1);
    if (state.smoothedScore === 0) state.smoothedScore = null;
  }

  state.history.push(state.smoothedScore);
  if (state.history.length > HISTORY_LEN * 2) {
    state.history.splice(0, state.history.length - HISTORY_LEN * 2);
  }

  updateScoreDisplay(state.smoothedScore);
  drawHistoryGraph();

  state.rafId = requestAnimationFrame(processFrame);
}

/* ─────────────────────────────────────────────────────────── */

/* ── Setup step handlers ───────────────────────────────────── */

async function handleShareTab() {
  dom.btnShare.disabled = true;
  setStatus('Requesting tab capture… (pick a tab in the dialog)');

  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: {
        width:  { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
      audio: false,
    });

    state.refStream       = stream;
    dom.refVideo.srcObject = stream;

    /* start playing so TF.js can read frames */
    try { await dom.refVideo.play(); } catch (_) { /* autoplay may handle it */ }
    await videoReady(dom.refVideo);

    /* react if the user stops sharing */
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      setStatus('Tab share ended. Click "Share Tab" to re-share.', true);
      dom.btnShare.disabled = false;
      dom.checkShare.classList.remove('visible');
      dom.stepShare.classList.remove('done');
      dom.btnStart.disabled = true;
      dom.btnCamera.disabled = true;
      state.refStream = null;
    });

    dom.checkShare.classList.add('visible');
    dom.stepShare.classList.add('done');
    dom.btnCamera.disabled = false;
    setStatus('Tab shared ✓ — now enable your camera.');

  } catch (err) {
    dom.btnShare.disabled = false;
    setStatus(
      err.name === 'NotAllowedError'
        ? 'Permission denied. Click "Share Tab" and choose a browser tab.'
        : `Could not share tab: ${err.message}`,
      true
    );
  }
}

async function handleEnableCamera() {
  dom.btnCamera.disabled = true;
  setStatus('Requesting camera access…');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:  { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    });

    state.camStream        = stream;
    dom.camVideo.srcObject = stream;

    try { await dom.camVideo.play(); } catch (_) { /* autoplay */ }
    await videoReady(dom.camVideo);

    dom.checkCamera.classList.add('visible');
    dom.stepCamera.classList.add('done');
    dom.btnStart.disabled = false;
    setStatus('Camera ready ✓ — click "Start Tracking"!');

  } catch (err) {
    dom.btnCamera.disabled = false;
    setStatus(
      err.name === 'NotAllowedError'
        ? 'Camera permission denied. Please allow camera access and try again.'
        : `Could not open camera: ${err.message}`,
      true
    );
  }
}

async function handleStart() {
  dom.btnStart.disabled = true;
  setStatus('Loading AI model…');
  show(dom.loadingOverlay);

  try {
    /* prefer WebGL for GPU-accelerated inference */
    await tf.setBackend('webgl');
    await tf.ready();

    state.detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );

  } catch (err) {
    hide(dom.loadingOverlay);
    dom.btnStart.disabled = false;
    setStatus(`Failed to load model: ${err.message}`, true);
    return;
  }

  hide(dom.loadingOverlay);

  /* transition to tracking view first so layout is available */
  hide(dom.setupPanel);
  show(dom.trackingPanel);

  /* size the history canvas now that the panel is visible */
  const hc   = dom.historyCanvas;
  const rect  = hc.getBoundingClientRect();
  hc.width    = Math.round(rect.width)  || 156;
  hc.height   = Math.round(rect.height) || 80;

  state.running       = true;
  state.smoothedScore = null;
  state.history       = [];

  state.rafId = requestAnimationFrame(processFrame);
}

function handleStop() {
  state.running = false;
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = null; }

  /* stop media tracks */
  state.refStream?.getTracks().forEach(t => t.stop());
  state.camStream?.getTracks().forEach(t => t.stop());
  state.refStream = null;
  state.camStream = null;

  /* free TF.js model */
  try { state.detector?.dispose(); } catch (_) { /* ignore */ }
  state.detector = null;

  /* reset video elements */
  dom.refVideo.srcObject = null;
  dom.camVideo.srcObject = null;

  /* reset UI controls */
  dom.btnShare.disabled  = false;
  dom.btnCamera.disabled = true;
  dom.btnStart.disabled  = true;
  dom.checkShare.classList.remove('visible');
  dom.checkCamera.classList.remove('visible');
  dom.stepShare.classList.remove('done');
  dom.stepCamera.classList.remove('done');
  setStatus('');

  show(dom.setupPanel);
  hide(dom.trackingPanel);
}

/* ─────────────────────────────────────────────────────────── */

/* ── Event bindings ────────────────────────────────────────── */
dom.btnShare .addEventListener('click', handleShareTab);
dom.btnCamera.addEventListener('click', handleEnableCamera);
dom.btnStart .addEventListener('click', handleStart);
dom.btnStop  .addEventListener('click', handleStop);
