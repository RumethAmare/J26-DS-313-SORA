// SORA C1 live transcript page. The browser records the microphone, sends
// 16 kHz float32 blocks to live_web.py over a WebSocket, and renders the
// events that come back (see the protocol in live_web.py).
"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  model: $("model"), toggle: $("toggle"), clear: $("clear"), dot: $("dot"), state: $("state"), level: $("level"),
  banner: $("banner"), transcript: $("transcript"), empty: $("empty"), tip: $("tip"),
  saved: $("saved"), dlJson: $("dl-json"), dlTokens: $("dl-tokens"),
  n: $("s-n"), audio: $("s-audio"), med: $("s-med"), max: $("s-max"), smodel: $("s-model"),
};
const LANG_CLASS = { SI: "si", EN: "en", OTHER: "other" };
const LANG_NAME = { SI: "Sinhala", EN: "English", OTHER: "Other (Tamil)" };
const WEAK_ALIGN = 0.05;   // below this the word's time is often misplaced (c1_timestamps_report.md)

let ws = null, audioCtx = null, stream = null, node = null;
let running = false, samplesSent = 0, lastDoc = null;
const tokensByUtt = new Map();
let partialRow = null;
const lags = [];

// ---------- helpers ----------
function setState(text, dot = "") {
  els.state.textContent = text;
  els.dot.className = "dot " + dot;
}
function banner(msg) {
  els.banner.textContent = msg || "";
  els.banner.style.display = msg ? "block" : "none";
}
function mmss(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
function nearBottom() {
  const m = els.transcript;
  return m.scrollHeight - m.scrollTop - m.clientHeight < 80;
}
function scrollDown(stick) {
  if (stick) els.transcript.scrollTop = els.transcript.scrollHeight;
}
function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b), h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}
function updateStats() {
  els.n.textContent = String(tokensByUtt.size);
  const m = median(lags);
  els.med.textContent = m == null ? "–" : `${m.toFixed(1)} s`;
  els.max.textContent = lags.length ? `${Math.max(...lags).toFixed(1)} s` : "–";
}

// ---------- transcript rendering ----------
function ensurePartialRow() {
  if (!partialRow) {
    partialRow = document.createElement("div");
    partialRow.className = "utt partial";
    partialRow.innerHTML = '<div class="time"></div><div class="words"></div>';
    els.transcript.appendChild(partialRow);
  }
  return partialRow;
}
function clearPartial() {
  if (partialRow) partialRow.remove();
  partialRow = null;
}
const EMPTY_START = els.empty.innerHTML;

// Removes the transcript from the page only; the saved session files and the
// download buttons stay. `note` replaces the start-up hint.
function clearTranscript(note) {
  clearPartial();
  els.transcript.querySelectorAll(".utt").forEach((r) => r.remove());
  tokensByUtt.clear();
  lags.length = 0;
  els.tip.style.display = "none";
  if (!running) {
    samplesSent = 0;
    els.audio.textContent = "0:00";
  }
  updateStats();
  els.empty.innerHTML = note || EMPTY_START;
  els.empty.style.display = "";
}
els.clear.onclick = () => clearTranscript();

function renderWords(container, tokens, text) {
  container.textContent = "";
  if (!tokens || !tokens.length) {
    const span = document.createElement("span");
    span.className = "notext";
    span.textContent = text || "(no text)";
    container.appendChild(span);
    return;
  }
  tokens.forEach((t, i) => {
    const span = document.createElement("span");
    span.className = "w " + (LANG_CLASS[t.lang] || "");
    if (t.align_score != null && t.align_score < WEAK_ALIGN) span.classList.add("weak");
    span.textContent = t.token;
    span.dataset.utt = t.utt_id;
    span.dataset.i = i;
    container.appendChild(span);
    container.appendChild(document.createTextNode(" "));
  });
}
function addUtterance(u, dur) {
  const stick = nearBottom();
  clearPartial();
  els.empty.style.display = "none";
  tokensByUtt.set(u.utt_id, u.tokens);
  const row = document.createElement("div");
  row.className = "utt";
  row.id = "u-" + u.utt_id;
  const time = document.createElement("div");
  time.className = "time";
  time.textContent = mmss(u.start);
  const words = document.createElement("div");
  words.className = "words";
  renderWords(words, u.tokens, u.text);
  const meta = document.createElement("div");
  meta.className = "meta";
  const bits = [`${dur.toFixed(1)} s speech`];
  if (u.end_reason === "pause") bits.push(`text ${u.latency_after_speech_s.toFixed(1)} s after the pause`);
  if (u.end_reason === "max") bits.push("cut at 25 s (long turn)");
  if (u.decode_action && u.decode_action !== "greedy") bits.push(`fixed: ${u.decode_action}`);
  meta.textContent = bits.join(" · ");
  row.append(time, words, meta);
  els.transcript.appendChild(row);
  if (u.end_reason === "pause") lags.push(u.latency_after_speech_s);
  updateStats();
  scrollDown(stick);
}
function updateTimes(uttId, tokens) {
  tokensByUtt.set(uttId, tokens);
  const row = document.getElementById("u-" + uttId);
  if (row) renderWords(row.querySelector(".words"), tokens);
}

// ---------- tooltip ----------
els.transcript.addEventListener("mousemove", (e) => {
  const w = e.target.closest(".w");
  if (!w) { els.tip.style.display = "none"; return; }
  const t = (tokensByUtt.get(w.dataset.utt) || [])[+w.dataset.i];
  if (!t) return;
  const f = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
  const rows = [
    ["language", `${LANG_NAME[t.lang] || t.lang} (${f(t.lang_confidence)} · ${t.lang_method})`],
    ["time", t.start == null ? "aligning…" : `${f(t.start)} – ${f(t.end)} s`],
    ["alignment", t.align_score == null ? "–" :
      `${f(t.align_score)}${t.align_score < WEAK_ALIGN ? " (weak: time may be off)" : ""}`],
    ["switch", t.switch ? "yes" : "no"],
  ];
  els.tip.innerHTML = "<table>" + rows.map(([k, v]) =>
    `<tr><td>${k}</td><td></td></tr>`).join("") + "</table>";
  els.tip.querySelectorAll("td:last-child").forEach((td, i) => { td.textContent = rows[i][1]; });
  els.tip.style.display = "block";
  const x = Math.min(e.clientX + 14, window.innerWidth - els.tip.offsetWidth - 8);
  const y = e.clientY + 18 + els.tip.offsetHeight > window.innerHeight ? e.clientY - els.tip.offsetHeight - 10 : e.clientY + 18;
  els.tip.style.left = x + "px";
  els.tip.style.top = y + "px";
});
els.transcript.addEventListener("mouseleave", () => { els.tip.style.display = "none"; });

// ---------- downloads ----------
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
els.dlJson.onclick = () => {
  if (!lastDoc) return;
  const s = lastDoc.session;
  download(`${s.model}_${s.started}.json`, JSON.stringify(lastDoc, null, 2), "application/json");
};
els.dlTokens.onclick = () => {
  if (!lastDoc) return;
  const s = lastDoc.session;
  const lines = lastDoc.utterances.flatMap((u) => u.tokens).map((t) => JSON.stringify(t)).join("\n") + "\n";
  download(`${s.model}_${s.started}.tokens.jsonl`, lines, "application/jsonl");
};

// ---------- microphone ----------
async function startMic() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  try {
    audioCtx = new AudioContext({ sampleRate: 16000 });
  } catch {
    audioCtx = new AudioContext();            // worklet downsamples
  }
  await audioCtx.audioWorklet.addModule("worklet.js");
  const src = audioCtx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(audioCtx, "mic-capture");
  node.port.onmessage = (e) => {
    const block = e.data;
    let sum = 0;
    for (let i = 0; i < block.length; i++) sum += block[i] * block[i];
    els.level.style.width = Math.min(100, Math.sqrt(sum / block.length) * 400) + "%";
    if (ws && ws.readyState === WebSocket.OPEN && running) {
      ws.send(block.buffer);
      samplesSent += block.length;
      els.audio.textContent = mmss(samplesSent / 16000);
    }
  };
  src.connect(node);
}
function stopMic() {
  if (node) node.disconnect();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  node = stream = audioCtx = null;
  els.level.style.width = "0";
}

// ---------- session ----------
function onEvent(ev) {
  switch (ev.type) {
    case "status":
      if (ev.state === "loading") setState(`loading ${ev.model}… (first time can take ~20 s)`);
      else if (ev.state === "busy") {
        banner("Another session is already running on this server (one GPU, one session). Stop it first.");
        finish();
      } else if (ev.state === "ready") {
        els.smodel.textContent = ev.run;
        startMic().then(() => {
          running = true;
          setState("listening", "ready");
          els.toggle.textContent = "Stop";
          els.toggle.disabled = false;
        }).catch((err) => {
          banner(`Microphone unavailable: ${err.message}. Open the page as http://localhost:8000 and allow the microphone.`);
          ws.send(JSON.stringify({ type: "stop" }));
        });
      }
      break;
    case "speech":
      if (running) setState(ev.on ? "speaking…" : "listening", ev.on ? "on" : "ready");
      break;
    case "partial": {
      const stick = nearBottom();
      const row = ensurePartialRow();
      row.querySelector(".words").textContent = ev.text ? ev.text + " …" : "…";
      els.empty.style.display = "none";
      scrollDown(stick);
      break;
    }
    case "discard":
      clearPartial();
      break;
    case "utterance":
      addUtterance(ev.utt, ev.dur);
      break;
    case "times":
      updateTimes(ev.utt_id, ev.tokens);
      break;
    case "saved":
      if (ev.doc) {
        lastDoc = ev.doc;
        els.dlJson.disabled = els.dlTokens.disabled = false;
        els.saved.textContent = `Saved on this laptop: ${ev.json} and ${ev.tokens}`;
        finish();
        clearTranscript(`Session ended: ${ev.doc.n_utterances} utterances saved. ` +
          "Use the buttons below to download it, or press <b>Start</b> for a new session.");
      } else {
        els.saved.textContent = "No speech captured; nothing saved.";
        finish();
        clearTranscript();
      }
      break;
    case "error":
      banner(ev.message);
      break;
  }
}

function start() {
  banner("");
  els.toggle.disabled = true;
  els.model.disabled = true;
  els.dlJson.disabled = els.dlTokens.disabled = true;
  lastDoc = null;
  samplesSent = 0;
  setState("connecting…");
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "start", model: els.model.value, partials: true }));
  ws.onmessage = (e) => onEvent(JSON.parse(e.data));
  ws.onclose = () => {
    if (running) banner("Lost the connection to the server. Is live_web.py still running?");
    finish();
  };
}
function stop() {
  running = false;
  stopMic();
  clearPartial();
  els.toggle.disabled = true;
  setState("finishing the last utterance and saving…");
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
}
function finish() {
  running = false;
  stopMic();
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
  els.toggle.textContent = "Start";
  els.toggle.disabled = false;
  els.model.disabled = false;
  setState("stopped");
}
els.toggle.onclick = () => (running ? stop() : start());

// ---------- start-up ----------
fetch("/api/models").then((r) => r.json()).then((d) => {
  for (const m of d.models) {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = m.available ? m.name : `${m.name} (not exported)`;
    o.disabled = !m.available;
    if (m.name === d.loaded) o.selected = true;
    els.model.appendChild(o);
  }
  els.smodel.textContent = d.loaded || "–";
  setState("ready — press Start");
  if (!window.isSecureContext) {
    banner("The browser only allows the microphone on http://localhost or https://. Open http://localhost:8000 on this laptop.");
  }
}).catch(() => {
  setState("server not reachable");
  banner("Can't reach live_web.py. Start it with: python c1/scripts/live_web.py");
});
