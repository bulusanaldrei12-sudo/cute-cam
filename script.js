import {
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

/* ---------- DOM ---------- */
const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const stage = $("stage");
const placeholder = $("placeholder");
const statusEl = $("status");
const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const chips = [...document.querySelectorAll("#chips li")];
const phoneBtn = $("phoneBtn");
const cameraSelect = $("cameraSelect");
const hdrBtn = $("hdrBtn");
let currentDeviceId = null; // null = default front camera
// Camera names that usually mean a phone/tablet: Apple Continuity Camera and common webcam apps
const PHONE_RE = /iphone|ipad|continuity|camo|epoccam|droidcam/i;

/* ---------- Effect definitions ---------- */
const MESSAGES = {
  smile: "😊 Smile detected — Leaf activated!",
  heart: "🫰 Finger heart detected — Heart Eyes activated!",
  wink: "😉 Wink detected — Sparkles activated!",
  peace: "✌️ Peace sign detected — Bunny Ears activated!",
  mouth: "😮 Open mouth detected — Bubbles activated!",
  kiss: "😘 Kiss detected — Flying Hearts activated!",
  hands: "🙌 Hands up detected — Flower Crown activated!",
  wave: "👋 Wave detected — Heart Trail activated!",
};
const KEYS = Object.keys(MESSAGES);
const level = {}; // smoothed 0..1 strength of each effect (drives fade/pop)
const active = {}; // raw on/off from detection
KEYS.forEach((k) => ((level[k] = 0), (active[k] = false)));

/* ---------- State ---------- */
let faceLandmarker = null;
let handLandmarker = null;
let stream = null;
let running = false;
let rafId = 0;
let lastVideoTime = -1;
let lastRender = performance.now();
let face = null; // smoothed face anchors (canvas pixels)
let lastHand = null; // last open-hand palm position (for wave trail)
let activeOrder = []; // most recently activated first
let uiSignature = "";
const wave = { hist: [], until: 0 };
const particles = [];
const timers = {};

/* ---------- Helpers ---------- */
const TAU = Math.PI * 2;
const lerp = (a, b, t) => a + (b - a) * t;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const rand = (a, b) => a + Math.random() * (b - a);
const easeOutBack = (t) => 1 + 2.70158 * Math.pow(t - 1, 3) + 1.70158 * Math.pow(t - 1, 2);
const toPx = (p) => ({ x: p.x * canvas.width, y: p.y * canvas.height });
const every = (name, ms, now, fn) => {
  if (now - (timers[name] || 0) > ms) {
    timers[name] = now;
    fn();
  }
};
const PINKS = ["#ff7fae", "#ffa6c9", "#ff9ecb", "#ffc2dd"];

function setStatus(text) {
  statusEl.textContent = text;
}

/* ---------- MediaPipe setup ---------- */
async function loadModels() {
  if (faceLandmarker && handLandmarker) return;
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const make = async (Cls, modelPath, extra) => {
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: modelPath, delegate },
      runningMode: "VIDEO",
      ...extra,
    });
    try {
      return await Cls.createFromOptions(fileset, opts("GPU"));
    } catch (e) {
      console.warn("GPU delegate failed, using CPU", e);
      return await Cls.createFromOptions(fileset, opts("CPU"));
    }
  };
  faceLandmarker = await make(FaceLandmarker, FACE_MODEL, {
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  handLandmarker = await make(HandLandmarker, HAND_MODEL, { numHands: 2 });
}

/* ---------- Camera start / stop ---------- */
// Ask the camera for continuous auto exposure / white balance / focus when it supports them.
async function tuneTrack(track) {
  try {
    const caps = track.getCapabilities?.() || {};
    const adv = {};
    for (const key of ["exposureMode", "whiteBalanceMode", "focusMode"]) {
      if (caps[key]?.includes("continuous")) adv[key] = "continuous";
    }
    if (Object.keys(adv).length) await track.applyConstraints({ advanced: [adv] });
  } catch (e) {
    console.warn("Could not tune camera", e);
  }
}

async function refreshCameraList() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const activeId = stream?.getVideoTracks()[0]?.getSettings().deviceId;
  cameraSelect.innerHTML = "";
  devices.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = d.deviceId;
    o.textContent = d.label || `Camera ${i + 1}`;
    o.selected = d.deviceId === activeId;
    cameraSelect.append(o);
  });
  return devices;
}

async function switchCamera(deviceId) {
  currentDeviceId = deviceId || null;
  if (running) stopCamera();
  await startCamera(currentDeviceId);
}

async function usePhoneCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("INSECURE");
    setStatus("Looking for your iPhone or iPad…");
    if (!stream) {
      // Camera names stay hidden until permission has been granted once
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach((t) => t.stop());
    }
    const devices = await refreshCameraList();
    const phone = devices.find((d) => PHONE_RE.test(d.label));
    if (!phone) {
      setStatus("No iPhone/iPad camera found. Unlock your iPhone near your Mac, then try again.");
      return;
    }
    await switchCamera(phone.deviceId);
  } catch (err) {
    console.error(err);
    setStatus(explainError(err));
  }
}

async function startCamera(deviceId = currentDeviceId) {
  startBtn.disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("INSECURE");
    }
    setStatus("Loading cute magic… ✨");
    await loadModels();
    setStatus("Waiting for camera permission…");
    const videoConstraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
    };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };
    else videoConstraints.facingMode = "user";
    stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
    await tuneTrack(stream.getVideoTracks()[0]);
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    placeholder.classList.add("hidden");
    stopBtn.disabled = false;
    running = true;
    lastVideoTime = -1;
    lastRender = performance.now();
    setStatus("Waiting for a cute gesture…");
    refreshCameraList().catch(() => {});
    loop();
  } catch (err) {
    console.error(err);
    if (err?.name === "OverconstrainedError") currentDeviceId = null;
    startBtn.disabled = false;
    setStatus(explainError(err));
    if (stream) stream.getTracks().forEach((t) => t.stop());
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.length = 0;
  face = null;
  wave.hist = [];
  wave.until = 0;
  activeOrder = [];
  KEYS.forEach((k) => ((level[k] = 0), (active[k] = false)));
  refreshUI();
  placeholder.classList.remove("hidden");
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Camera is off");
}

function explainError(err) {
  const n = err?.name || err?.message;
  if (n === "INSECURE") return "Camera needs https or localhost. Open the page with Live Server.";
  if (n === "NotAllowedError") return "Camera blocked. Allow camera access in the address bar, then try again.";
  if (n === "NotFoundError") return "No camera found. Plug one in and try again.";
  if (n === "OverconstrainedError") return "That camera isn't available right now. Press Start Camera to use the default one.";
  if (n === "NotReadableError") return "Camera is busy. Close other apps using it, then try again.";
  return "Something went wrong: " + (err?.message || err) + " (check the console)";
}

/* ---------- Main loop ---------- */
function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  const now = performance.now();
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    detect(now);
  }
  render(now);
}

/* ---------- Detection ---------- */
function updateFace(lm) {
  const P = (i) => toPx(lm[i]);
  const mid = (i, j) => {
    const a = P(i), b = P(j);
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  };
  const l = P(234), r = P(454);
  const next = {
    top: P(10), // forehead top
    eyeL: mid(33, 133),
    eyeR: mid(362, 263),
    mouth: mid(13, 14),
    w: dist(l, r),
    a: Math.atan2(r.y - l.y, r.x - l.x), // head roll
  };
  if (!face) {
    face = next;
    return;
  }
  const k = 0.35; // smoothing: lower = smoother, higher = snappier
  for (const key of ["top", "eyeL", "eyeR", "mouth"]) {
    face[key].x = lerp(face[key].x, next[key].x, k);
    face[key].y = lerp(face[key].y, next[key].y, k);
  }
  face.w = lerp(face.w, next.w, k);
  face.a = lerp(face.a, next.a, k);
}

// A finger is "extended" when its tip is clearly farther from the wrist than its middle joint.
const ext = (h, tip, pip) => dist(h[0], h[tip]) > dist(h[0], h[pip]) * 1.12;

function trackWave(now, hand) {
  if (!hand) {
    wave.hist = [];
    return;
  }
  wave.hist.push({ t: now, x: hand[9].x });
  while (wave.hist.length && now - wave.hist[0].t > 900) wave.hist.shift();
  let dir = 0, reversals = 0, ref = wave.hist[0].x;
  for (const p of wave.hist) {
    const d = p.x - ref;
    if (Math.abs(d) > canvas.width * 0.03) {
      const s = Math.sign(d);
      if (dir && s !== dir) reversals++;
      dir = s;
      ref = p.x;
    }
  }
  if (reversals >= 2) wave.until = now + 500;
}

function detect(now) {
  const T = {};
  KEYS.forEach((k) => (T[k] = false));

  /* Face + blendshapes */
  const f = faceLandmarker.detectForVideo(video, now);
  const hasFace = f.faceLandmarks.length > 0;
  if (hasFace) {
    updateFace(f.faceLandmarks[0]);
    const bs = {};
    for (const c of f.faceBlendshapes[0].categories) bs[c.categoryName] = c.score;
    const smile = (bs.mouthSmileLeft + bs.mouthSmileRight) / 2;
    const blinkL = bs.eyeBlinkLeft, blinkR = bs.eyeBlinkRight;
    T.smile = smile > 0.5;
    T.wink = Math.abs(blinkL - blinkR) > 0.3 && Math.max(blinkL, blinkR) > 0.5;
    T.mouth = bs.jawOpen > 0.4 && bs.mouthPucker < 0.4;
    T.kiss = bs.mouthPucker > 0.45 && bs.jawOpen < 0.35;
  }

  /* Hands */
  const h = handLandmarker.detectForVideo(video, now);
  const hands = h.landmarks.map((lm) => lm.map(toPx));
  let openHand = null;
  for (const hd of hands) {
    const size = dist(hd[0], hd[9]); // palm size: makes rules scale-independent
    const othersCurled = !ext(hd, 12, 10) && !ext(hd, 16, 14) && !ext(hd, 20, 18);

    // Finger heart: thumb tip + index tip touching, other fingers curled, index not fully folded
    const fingerHeart =
      dist(hd[4], hd[8]) < size * 0.45 && othersCurled && dist(hd[0], hd[8]) > size * 1.1;
    // Peace: index + middle up and apart, ring + pinky curled
    const peace =
      ext(hd, 8, 6) && ext(hd, 12, 10) && !ext(hd, 16, 14) && !ext(hd, 20, 18) &&
      dist(hd[8], hd[12]) > size * 0.35;
    const isOpen = ext(hd, 8, 6) && ext(hd, 12, 10) && ext(hd, 16, 14) && ext(hd, 20, 18);

    if (hasFace && fingerHeart) T.heart = true;
    if (hasFace && peace) T.peace = true;
    if (isOpen && !openHand) openHand = hd;
  }

  /* Wave (open hand moving left-right repeatedly) */
  trackWave(now, openHand);
  if (openHand) lastHand = { x: openHand[9].x, y: openHand[9].y };
  T.wave = now < wave.until;

  /* Both hands raised above the eye line */
  if (hasFace && hands.length >= 2) {
    const eyeY = (face.eyeL.y + face.eyeR.y) / 2;
    T.hands = hands.slice(0, 2).every((hd) => hd[9].y < eyeY + face.w * 0.1);
  }

  /* Commit + keep a "most recent first" list for the status bar */
  for (const k of KEYS) {
    if (T[k] && !active[k]) activeOrder.unshift(k);
    if (!T[k] && active[k]) activeOrder = activeOrder.filter((x) => x !== k);
    active[k] = T[k];
  }
}

/* ---------- Drawing primitives ---------- */
function heartPath(x, y, s) {
  ctx.beginPath();
  ctx.moveTo(x, y + s * 0.35);
  ctx.bezierCurveTo(x - s * 0.7, y - s * 0.05, x - s * 0.35, y - s * 0.6, x, y - s * 0.25);
  ctx.bezierCurveTo(x + s * 0.35, y - s * 0.6, x + s * 0.7, y - s * 0.05, x, y + s * 0.35);
  ctx.closePath();
}

function drawHeart(x, y, s, color = "#ff7fae", glow = true) {
  ctx.save();
  if (glow) {
    ctx.shadowColor = "rgba(255,105,170,0.9)";
    ctx.shadowBlur = s * 0.6;
  }
  heartPath(x, y, s);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(255,255,255,0.7)"; // little shine
  ctx.beginPath();
  ctx.ellipse(x - s * 0.28, y - s * 0.2, s * 0.09, s * 0.05, -0.6, 0, TAU);
  ctx.fill();
  ctx.restore();
}

function drawSparkle(x, y, s, rot = 0, color = "#fff") {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  ctx.shadowColor = "rgba(255,200,230,1)";
  ctx.shadowBlur = s * 0.8;
  ctx.fillStyle = color;
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const a = (i * Math.PI) / 4 - Math.PI / 2;
    const r = i % 2 ? s * 0.22 : s;
    ctx[i ? "lineTo" : "moveTo"](Math.cos(a) * r, Math.sin(a) * r);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawBubble(x, y, r) {
  ctx.save();
  const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
  g.addColorStop(0, "rgba(255,255,255,0.35)");
  g.addColorStop(1, "rgba(190,225,255,0.12)");
  ctx.fillStyle = g;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = Math.max(1.5, r * 0.08);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, TAU);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.beginPath();
  ctx.arc(x, y, r * 0.7, Math.PI * 1.1, Math.PI * 1.45);
  ctx.stroke();
  ctx.restore();
}

function drawFlower(x, y, r, color) {
  ctx.fillStyle = color;
  for (let i = 0; i < 5; i++) {
    const a = (i * TAU) / 5;
    ctx.beginPath();
    ctx.arc(x + Math.cos(a) * r * 0.7, y + Math.sin(a) * r * 0.7, r * 0.55, 0, TAU);
    ctx.fill();
  }
  ctx.fillStyle = "#ffe27a";
  ctx.beginPath();
  ctx.arc(x, y, r * 0.45, 0, TAU);
  ctx.fill();
}

/* ---------- Face-anchored effects (drawn in head-local coordinates) ---------- */
// Origin = forehead top, x axis follows head tilt, units are pixels (face.w = head width).
function inHead(alpha, pop, fn) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(face.top.x, face.top.y);
  ctx.rotate(face.a);
  ctx.scale(pop, pop);
  fn(face.w);
  ctx.restore();
}

function drawLeaf(t) {
  const L = level.smile;
  inHead(L, easeOutBack(Math.min(1, L)), (s) => {
    ctx.translate(0, -0.12 * s);
    const leaf = (rot, scale) => {
      ctx.save();
      ctx.rotate(rot + Math.sin(t * 2.2) * 0.07);
      const len = 0.3 * s * scale;
      const g = ctx.createLinearGradient(0, 0, 0, -len);
      g.addColorStop(0, "#7fd69a");
      g.addColorStop(1, "#c4f2b4");
      ctx.fillStyle = g;
      ctx.strokeStyle = "#58b57a";
      ctx.lineWidth = s * 0.012;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(len * 0.7, -len * 0.2, len * 0.6, -len * 0.9, 0, -len);
      ctx.bezierCurveTo(-len * 0.6, -len * 0.9, -len * 0.7, -len * 0.2, 0, 0);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, -len * 0.8);
      ctx.stroke();
      ctx.restore();
    };
    leaf(-0.45, 0.8);
    leaf(0.45, 1);
    const tw = 0.6 + 0.4 * Math.sin(t * 5);
    drawSparkle(0.24 * s, -0.3 * s + Math.sin(t * 2) * 0.02 * s, s * 0.05 * tw, t);
    drawSparkle(-0.2 * s, -0.18 * s, s * 0.03 * (1.1 - tw * 0.5), -t);
  });
}

function drawBunnyEars(t) {
  const L = level.peace;
  inHead(L, easeOutBack(Math.min(1, L)), (s) => {
    const bounce = -Math.abs(Math.sin(t * 5)) * 0.035 * s;
    for (const side of [-1, 1]) {
      ctx.save();
      ctx.translate(side * 0.2 * s, -0.04 * s + bounce);
      ctx.rotate(side * 0.2 + Math.sin(t * 5 + (side > 0 ? 0 : 1.2)) * 0.07);
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "#ffd3e4";
      ctx.lineWidth = s * 0.018;
      ctx.beginPath();
      ctx.ellipse(0, -0.3 * s, 0.1 * s, 0.3 * s, 0, 0, TAU);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffc2d9";
      ctx.beginPath();
      ctx.ellipse(0, -0.28 * s, 0.05 * s, 0.21 * s, 0, 0, TAU);
      ctx.fill();
      ctx.restore();
    }
  });
}

function drawCrown(t) {
  const L = level.hands;
  inHead(L, easeOutBack(Math.min(1, L)), (s) => {
    const colors = ["#ffb7d0", "#fff", "#d9ccff", "#ffd8a8", "#bfe9ff"];
    const n = 9;
    ctx.fillStyle = "#8fd6a0";
    for (let i = 0; i < n; i++) {
      const u = (i / (n - 1)) * 2 - 1; // -1..1 across the head
      const x = u * 0.5 * s;
      const y = -0.03 * s + u * u * 0.2 * s;
      ctx.beginPath(); // leaf behind each flower
      ctx.ellipse(x + 0.04 * s, y + 0.03 * s, 0.05 * s, 0.022 * s, 0.6, 0, TAU);
      ctx.fill();
      drawFlower(x, y, s * (i % 2 ? 0.055 : 0.07), colors[i % colors.length]);
    }
    for (let i = 0; i < 3; i++) {
      const tw = 0.5 + 0.5 * Math.sin(t * 4 + i * 2);
      drawSparkle((i - 1) * 0.28 * s, -0.16 * s - (i % 2) * 0.05 * s, s * 0.04 * tw, t + i);
    }
  });
}

function drawHeartEyes(t) {
  const L = level.heart;
  const pop = easeOutBack(Math.min(1, L));
  ctx.save();
  ctx.globalAlpha = L;
  for (const eye of [face.eyeL, face.eyeR]) {
    const beat = 1 + 0.08 * Math.sin(t * 8);
    drawHeart(eye.x, eye.y, face.w * 0.2 * pop * beat, "#ff6fa5");
    for (let i = 0; i < 3; i++) {
      const a = t * 2 + (i * TAU) / 3;
      const r = face.w * 0.14;
      drawHeart(eye.x + Math.cos(a) * r, eye.y + Math.sin(a) * r * 0.8, face.w * 0.05 * pop, "#ffb0cf");
    }
    drawSparkle(eye.x + face.w * 0.1, eye.y - face.w * 0.11, face.w * 0.035 * (0.7 + 0.3 * Math.sin(t * 6)), t);
  }
  ctx.restore();
}

/* ---------- Particles (wink sparkles, bubbles, kiss hearts, wave trail) ---------- */
function spawn(type, x, y, o = {}) {
  particles.push({
    type, x, y,
    vx: o.vx ?? 0, vy: o.vy ?? -0.5,
    life: 0, max: o.max ?? 1.5,
    size: o.size ?? 10, rot: rand(-0.5, 0.5), vr: o.vr ?? 0,
    color: o.color ?? "#fff",
  });
}

function faceCenter() {
  return {
    x: (face.eyeL.x + face.eyeR.x) / 2,
    y: (face.eyeL.y + face.eyeR.y) / 2 + face.w * 0.18,
  };
}

function spawnEffects(now) {
  if (face) {
    const c = faceCenter();
    if (active.wink) {
      every("wink", 70, now, () => {
        const a = rand(0, TAU), r = face.w * rand(0.5, 0.8);
        spawn("sparkle", c.x + Math.cos(a) * r, c.y + Math.sin(a) * r * 1.1, {
          vx: rand(-0.3, 0.3), vy: rand(-0.4, 0.1), max: 0.9,
          size: face.w * rand(0.04, 0.08), vr: rand(-0.08, 0.08),
          color: Math.random() > 0.5 ? "#fff" : "#fff3a8",
        });
      });
    }
    if (active.mouth) {
      every("bubble", 150, now, () => {
        const a = rand(0, TAU), r = face.w * rand(0.4, 0.75);
        spawn("bubble", c.x + Math.cos(a) * r, c.y + Math.sin(a) * r, {
          vx: rand(-0.2, 0.2), vy: rand(-0.7, -0.35), max: 2.4,
          size: face.w * rand(0.03, 0.08),
        });
      });
    }
    if (active.kiss) {
      every("kiss", 200, now, () => {
        spawn("heart", face.mouth.x + rand(-6, 6), face.mouth.y, {
          vx: rand(-0.5, 0.5), vy: rand(-1.4, -0.9), max: 1.6,
          size: face.w * rand(0.05, 0.09), vr: rand(-0.02, 0.02),
          color: PINKS[(Math.random() * PINKS.length) | 0],
        });
      });
    }
  }
  if (active.wave && lastHand) {
    every("wave", 55, now, () => {
      spawn("heart", lastHand.x + rand(-10, 10), lastHand.y + rand(-10, 10), {
        vx: rand(-0.2, 0.2), vy: rand(-0.4, -0.1), max: 0.9,
        size: canvas.width * rand(0.02, 0.035),
        color: PINKS[(Math.random() * PINKS.length) | 0],
      });
    });
  }
}

function drawParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life += dt;
    if (p.life >= p.max) {
      particles.splice(i, 1);
      continue;
    }
    p.x += p.vx * dt * 60;
    p.y += p.vy * dt * 60;
    p.rot += p.vr * dt * 60;
    if (p.type === "bubble") p.x += Math.sin(p.life * 3 + p.size) * 0.4; // gentle wobble
    const fadeIn = Math.min(1, p.life / 0.15);
    const fadeOut = 1 - Math.pow(p.life / p.max, 2);
    const pop = easeOutBack(Math.min(1, p.life / 0.25));
    ctx.save();
    ctx.globalAlpha = Math.max(0, fadeIn * fadeOut);
    if (p.type === "heart") {
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      drawHeart(0, 0, p.size * pop, p.color);
    } else if (p.type === "sparkle") {
      drawSparkle(p.x, p.y, p.size * pop * (0.6 + 0.4 * Math.sin(p.life * 18)), p.rot, p.color);
    } else {
      drawBubble(p.x, p.y, p.size * pop);
    }
    ctx.restore();
  }
}

/* ---------- Render ---------- */
function render(now) {
  const dt = Math.min(0.05, (now - lastRender) / 1000);
  lastRender = now;
  const t = now / 1000;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Ease each effect's strength toward 1 (on) or 0 (off) -> smooth fade in/out + pop
  for (const k of KEYS) {
    const rate = active[k] ? 10 : 5;
    level[k] = lerp(level[k], active[k] ? 1 : 0, 1 - Math.exp(-dt * rate));
    if (!active[k] && level[k] < 0.01) level[k] = 0;
  }

  spawnEffects(now);

  if (face) {
    if (level.heart > 0) drawHeartEyes(t);
    if (level.hands > 0) drawCrown(t);
    if (level.peace > 0) drawBunnyEars(t);
    if (level.smile > 0) drawLeaf(t);
  }
  drawParticles(dt);
  refreshUI();
}

/* ---------- UI ---------- */
function refreshUI() {
  const sig = activeOrder.join(",");
  if (sig === uiSignature) return;
  uiSignature = sig;
  chips.forEach((c) => c.classList.toggle("active", active[c.dataset.fx]));
  if (running) {
    setStatus(activeOrder.length ? MESSAGES[activeOrder[0]] : "Waiting for a cute gesture…");
  }
}

startBtn.addEventListener("click", () => startCamera());
stopBtn.addEventListener("click", stopCamera);
phoneBtn.addEventListener("click", usePhoneCamera);
cameraSelect.addEventListener("change", () => switchCamera(cameraSelect.value));

hdrBtn.addEventListener("click", () => {
  const on = stage.classList.toggle("hdr");
  hdrBtn.setAttribute("aria-pressed", String(on));
  hdrBtn.textContent = on ? "✨ HDR: On" : "✨ HDR: Off";
});

// Plugging in / waking up a phone camera updates the list automatically
navigator.mediaDevices?.addEventListener("devicechange", () => {
  if (running) refreshCameraList().catch(() => {});
});
