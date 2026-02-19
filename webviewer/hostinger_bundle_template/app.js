let bundle = null;
let currentIndex = 0;
let timer = null;

const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const statsEl = document.getElementById("stats");
const frameInput = document.getElementById("frame");
const fpsInput = document.getElementById("fps");

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function frameCount() {
  return bundle?.frames?.length ?? 0;
}

function drawFrame(index) {
  if (!bundle || frameCount() === 0) {
    return;
  }

  currentIndex = clamp(index, 0, frameCount() - 1);
  frameInput.value = String(currentIndex);

  const meta = bundle.meta;
  const frame = bundle.frames[currentIndex];
  const gridWidth = Math.max(1, meta.grid_width || 1);
  const cell = canvas.width / gridWidth;
  const maxResource = Math.max(1, meta.max_resource || 1);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const resources = frame.resources || [];
  for (let i = 0; i < resources.length; i += 1) {
    const value = Number(resources[i] || 0);
    const brightness = clamp(Math.floor((value / maxResource) * 255), 0, 255);
    const x = (i % gridWidth) * cell;
    const y = Math.floor(i / gridWidth) * cell;
    ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
    ctx.fillRect(x, y, cell + 0.25, cell + 0.25);
  }

  ctx.fillStyle = "rgba(35, 180, 255, 0.92)";
  const radius = Math.max(1, Math.min(3, cell * 0.45));
  const positions = frame.positions || [];
  for (const pos of positions) {
    const p = Number(pos);
    const x = (p % gridWidth) * cell + cell * 0.5;
    const y = Math.floor(p / gridWidth) * cell + cell * 0.5;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  statsEl.textContent = `t=${frame.t} | frame ${currentIndex + 1}/${frameCount()} | population=${frame.population} | avg resources=${frame.avg_individual_resources}`;
}

function stopPlayback() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function startPlayback() {
  stopPlayback();
  const fps = clamp(Number(fpsInput.value || 10), 1, 60);
  timer = setInterval(() => {
    const next = (currentIndex + 1) % frameCount();
    drawFrame(next);
  }, 1000 / fps);
}

async function loadBundle() {
  const response = await fetch("./data/dashboard_data.json");
  if (!response.ok) {
    throw new Error(`Failed to load data: ${response.status}`);
  }
  bundle = await response.json();
  const max = Math.max(0, frameCount() - 1);
  frameInput.max = String(max);
  drawFrame(0);
}

document.getElementById("btnPlay").addEventListener("click", startPlayback);
document.getElementById("btnPause").addEventListener("click", stopPlayback);
document.getElementById("btnPrev").addEventListener("click", () => drawFrame(currentIndex - 1));
document.getElementById("btnNext").addEventListener("click", () => drawFrame(currentIndex + 1));
frameInput.addEventListener("input", (event) => drawFrame(Number(event.target.value || 0)));
fpsInput.addEventListener("input", () => {
  if (timer) {
    startPlayback();
  }
});

loadBundle().catch((error) => {
  statsEl.textContent = `Could not load replay bundle: ${error.message}`;
});
