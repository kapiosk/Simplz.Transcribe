// --- Microphone streaming ---------------------------------------------------

const micBtn = document.getElementById("micBtn");
const micStatus = document.getElementById("micStatus");
const micTranscript = document.getElementById("micTranscript");

let mic = null; // { ws, audioContext, stream } while recording

micBtn.addEventListener("click", () => (mic ? stopMic() : startMic().catch(showMicError)));

async function startMic() {
  micTranscript.textContent = "";
  setStatus(micStatus, "requesting microphone…");

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true },
  });

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/transcribe`);

  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ type: "start" }));
    setStatus(micStatus, "listening… (transcript follows with a few seconds of delay on CPU)");
  });
  ws.addEventListener("message", (e) => handleEvent(JSON.parse(e.data), micTranscript, micStatus));
  ws.addEventListener("error", () => showMicError(new Error("WebSocket error")));
  ws.addEventListener("close", () => {
    if (mic && mic.ws === ws) teardownMic(); // server-side close
  });

  const audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("pcm-worklet.js");
  const source = audioContext.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(audioContext, "pcm-processor");
  worklet.port.onmessage = (e) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(e.data.buffer);
  };
  source.connect(worklet);

  mic = { ws, audioContext, stream };
  micBtn.textContent = "Stop";
  micBtn.classList.add("recording");
}

function stopMic() {
  if (!mic) return;
  const { ws } = mic;
  teardownMic();
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" })); // final transcript still arrives on this socket
    setStatus(micStatus, "finishing…");
  }
}

function teardownMic() {
  if (!mic) return;
  mic.stream.getTracks().forEach((t) => t.stop());
  mic.audioContext.close();
  mic = null;
  micBtn.textContent = "Start recording";
  micBtn.classList.remove("recording");
}

function showMicError(err) {
  teardownMic();
  setStatus(micStatus, err.message, true);
}

// --- File upload --------------------------------------------------------------

const fileInput = document.getElementById("fileInput");
const fileStatus = document.getElementById("fileStatus");
const fileTranscript = document.getElementById("fileTranscript");

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileTranscript.textContent = "";
  setStatus(fileStatus, `uploading & transcribing ${file.name}…`);
  fileInput.disabled = true;
  try {
    const response = await fetch("/api/transcribe-file", { method: "POST", body: file });
    if (!response.ok) throw new Error(`upload failed: HTTP ${response.status}`);
    await readSse(response, (evt) => handleEvent(evt, fileTranscript, fileStatus));
  } catch (err) {
    setStatus(fileStatus, err.message, true);
  } finally {
    fileInput.disabled = false;
    fileInput.value = "";
  }
});

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    pending += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = pending.indexOf("\n\n")) >= 0) {
      const frame = pending.slice(0, idx);
      pending = pending.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
      }
    }
  }
}

// --- Shared -------------------------------------------------------------------

function handleEvent(evt, transcriptEl, statusEl) {
  if (evt.type === "partial") {
    transcriptEl.textContent += evt.text; // deltas are append-only
  } else if (evt.type === "final") {
    transcriptEl.textContent = evt.text;
    setStatus(statusEl, "done");
    if (statusEl === micStatus) teardownMic();
  } else if (evt.type === "error") {
    setStatus(statusEl, evt.text, true);
    if (statusEl === micStatus) teardownMic();
  }
}

function setStatus(el, text, isError = false) {
  el.textContent = text;
  el.classList.toggle("error", isError);
}
