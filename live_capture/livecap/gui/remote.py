"""Phone-friendly local control server for the live GUI."""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Capture Remote</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      --bg: #12151a;
      --panel: #20252d;
      --line: #343b47;
      --text: #f3f6fa;
      --muted: #9faab8;
      --blue: #2f75bd;
      --green: #2d8a51;
      --red: #b8323b;
      --amber: #c28b2c;
    }
    body { margin: 0; background: var(--bg); color: var(--text); }
    main { max-width: 620px; margin: 0 auto; padding: 16px; }
    header { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin: 4px 0 12px; }
    h1 { font-size: 24px; margin: 0; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 3px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; margin: 12px 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; }
    .metric { background: #151922; border: 1px solid #303845; border-radius: 10px; padding: 10px; }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 21px; font-weight: 800; margin-top: 3px; }
    button, input { font: inherit; border-radius: 9px; border: 1px solid #48515f; padding: 12px; }
    button { color: var(--text); background: var(--blue); font-weight: 800; min-height: 48px; }
    button.stop { background: var(--red); }
    button.secondary { background: #3a414d; }
    button.good { background: var(--green); }
    button.active { background: var(--amber); color: #16120a; border-color: #e3b85b; }
    button:disabled { opacity: 0.55; }
    label { color: var(--muted); font-size: 12px; display: block; margin: 0 0 5px; }
    input { box-sizing: border-box; width: 100%; background: #101319; color: var(--text); }
    .row { display: flex; gap: 10px; align-items: end; }
    .row .grow { flex: 1; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { border: 1px solid var(--line); border-radius: 999px; padding: 7px 10px; color: var(--muted); font-size: 13px; background: #171c24; }
    .chip.on { color: #dff7e8; border-color: #386d4c; background: #163522; }
    .chip.warn { color: #ffe8b8; border-color: #6e5628; background: #372816; }
    .bar { height: 9px; border-radius: 999px; background: #11151c; overflow: hidden; border: 1px solid #303845; }
    .bar span { display: block; height: 100%; width: 0%; background: var(--blue); }
    .msg { color: var(--muted); font-size: 13px; min-height: 18px; margin-top: 8px; }
    .error { color: #ffb3b8; }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Live Capture</h1>
      <div class="sub">Phone remote</div>
    </div>
    <button class="secondary" onclick="refresh()">Refresh</button>
  </header>

  <div class="chips">
    <div id="captureChip" class="chip">Capture idle</div>
    <div id="recordChip" class="chip">Not recording</div>
    <div id="modeChip" class="chip">Mode --</div>
    <div id="calibChip" class="chip">3D --</div>
  </div>

  <div class="card">
    <label class="muted" for="pin">PIN</label>
    <div class="row">
      <div class="grow"><input id="pin" inputmode="numeric" autocomplete="off" placeholder="Remote PIN"></div>
      <button class="secondary" onclick="savePin()">Save</button>
    </div>
  </div>

  <div class="card grid">
    <button onclick="cmd('start_capture')">Start Capture</button>
    <button class="secondary" onclick="cmd('stop_capture')">Stop Capture</button>
    <button class="good" onclick="cmd('arm_recording')">Arm Recording</button>
    <button class="stop" onclick="cmd('stop_save')">Stop + Save</button>
    <button onclick="cmd('process_latest')">Process Latest</button>
    <button id="selfPacedBtn" class="secondary" onclick="cmd('toggle_self_paced')">Self-Paced</button>
    <button class="secondary" onclick="cmd('treadmill_stop')">Treadmill Stop</button>
  </div>

  <div class="card">
    <div class="metrics">
      <div class="metric"><div class="label">Current</div><div id="speedCurrent" class="value">--</div></div>
      <div class="metric"><div class="label">Left</div><div id="speedLeft" class="value">--</div></div>
      <div class="metric"><div class="label">Right</div><div id="speedRight" class="value">--</div></div>
    </div>
    <div class="sub" id="targetReadout">Target -- m/s | Incline -- deg</div>
  </div>

  <div class="card">
    <div class="sub" style="margin-bottom:10px;">Fixed treadmill mode</div>
    <div class="grid">
      <div><label for="velocity">Velocity (m/s)</label><input id="velocity" type="number" step="0.1" value="0.0"></div>
      <div><label for="acceleration">Acceleration (m/s2)</label><input id="acceleration" type="number" step="0.1" value="0.5"></div>
      <div><label for="incline">Incline (deg)</label><input id="incline" type="number" step="0.5" value="0.0"></div>
      <button onclick="setFixed()">Set Fixed</button>
    </div>
  </div>

  <div class="card">
    <div class="sub">Processing</div>
    <div class="bar"><span id="processingBar"></span></div>
    <div id="processingText" class="msg">Idle</div>
    <div class="sub" style="margin-top:10px;">Latest trial</div>
    <div id="trialText" class="msg">--</div>
    <div id="message" class="msg"></div>
  </div>
</main>
<script>
const pinEl = document.getElementById('pin');
pinEl.value = localStorage.getItem('livecap_pin') || '';
function pin() { return pinEl.value.trim(); }
function savePin() { localStorage.setItem('livecap_pin', pin()); refresh(); }
async function api(path, opts={}) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) throw new Error(data.error || res.statusText);
  return data;
}
async function cmd(command) {
  try {
    localStorage.setItem('livecap_pin', pin());
    const data = await api('/api/command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pin: pin(), command})
    });
    showMessage(data.message || command);
    update(data.status || data);
    setTimeout(refresh, 300);
  } catch (e) { showMessage(e.message, true); }
}
async function setFixed() {
  try {
    localStorage.setItem('livecap_pin', pin());
    const data = await api('/api/command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        pin: pin(),
        command: 'set_fixed',
        velocity: Number(document.getElementById('velocity').value || 0),
        acceleration: Number(document.getElementById('acceleration').value || 0.5),
        incline: Number(document.getElementById('incline').value || 0)
      })
    });
    showMessage(data.message || 'Fixed mode set');
    update(data.status || data);
    setTimeout(refresh, 300);
  } catch (e) { showMessage(e.message, true); }
}
async function refresh() {
  try { update(await api('/api/status?pin=' + encodeURIComponent(pin()))); }
  catch (e) { showMessage(e.message, true); }
}
function fmt(v, unit='') {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) + unit : '--';
}
function chip(id, text, on=false, warn=false) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = 'chip' + (on ? ' on' : '') + (warn ? ' warn' : '');
}
function update(data) {
  if (!data || data.ok === false) {
    showMessage((data && data.error) || 'No status', true);
    return;
  }
  chip('captureChip', data.capture_running ? 'Capture running' : 'Capture idle', !!data.capture_running);
  chip('recordChip', data.recording ? 'Recording' : 'Not recording', !!data.recording, !!data.recording);
  chip('modeChip', data.self_paced ? 'Self-Paced ON' : (data.treadmill_mode || 'Mode --'), !!data.self_paced);
  chip('calibChip', data.calibrated_3d ? '3D calibrated' : '3D not calibrated', !!data.calibrated_3d, !data.calibrated_3d);
  document.getElementById('selfPacedBtn').className = data.self_paced ? 'active' : 'secondary';
  document.getElementById('selfPacedBtn').textContent = data.self_paced ? 'Self-Paced: ON' : 'Self-Paced: OFF';
  document.getElementById('speedCurrent').textContent = fmt(data.treadmill_speed_m_s);
  document.getElementById('speedLeft').textContent = fmt(data.treadmill_left_m_s);
  document.getElementById('speedRight').textContent = fmt(data.treadmill_right_m_s);
  document.getElementById('targetReadout').textContent =
    `Target ${fmt(data.treadmill_target_m_s, ' m/s')} | Incline ${fmt(data.treadmill_incline_deg, ' deg')}`;
  const proc = data.processing || {};
  const pct = Math.round(Number(proc.progress || 0) * 100);
  document.getElementById('processingBar').style.width = pct + '%';
  document.getElementById('processingText').textContent = `${pct}% ${proc.message || 'Idle'}`;
  const trial = data.latest_trial;
  document.getElementById('trialText').textContent = trial
    ? `${trial.recording_name || trial.path} | ${trial.status || '--'} | ${trial.total_steps || 0} steps`
    : '--';
}
function showMessage(text, isError=false) {
  const el = document.getElementById('message');
  el.textContent = text || '';
  el.className = 'msg' + (isError ? ' error' : '');
}
setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""


def local_urls(port: int) -> list[str]:
    urls = [f"http://127.0.0.1:{port}"]
    seen = {"127.0.0.1"}
    try:
        host = socket.gethostname()
        for _, _, _, _, sockaddr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
            ip = sockaddr[0]
            if ":" not in ip and not ip.startswith("127.") and ip not in seen:
                seen.add(ip)
                urls.append(f"http://{ip}:{port}")
    except Exception:
        pass
    return urls


class RemoteControlServer:
    def __init__(self, app, host: str, port: int, pin: str):
        self.app = app
        self.host = host
        self.port = int(port)
        self.pin = str(pin or "")
        self.httpd = None
        self.thread = None

    def start(self):
        if self.httpd is not None:
            return
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "LiveCaptureRemote/1.0"

            def log_message(self, fmt, *args):
                print(f"[remote] {self.address_string()} - {fmt % args}")

            def _send(self, code, payload, content_type="application/json"):
                body = (payload if isinstance(payload, bytes)
                        else json.dumps(payload).encode("utf-8"))
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self, supplied):
                return bool(outer.pin) and str(supplied or "") == outer.pin

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/status":
                    pin = parse_qs(parsed.query).get("pin", [""])[0]
                    if not self._authorized(pin):
                        self._send(403, {"ok": False, "error": "Bad or missing PIN"})
                        return
                    self._send(200, outer.app.remote_status())
                    return
                self._send(404, {"ok": False, "error": "Not found"})

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path != "/api/command":
                    self._send(404, {"ok": False, "error": "Not found"})
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                try:
                    data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                except Exception:
                    self._send(400, {"ok": False, "error": "Invalid JSON"})
                    return
                if not self._authorized(data.get("pin")):
                    self._send(403, {"ok": False, "error": "Bad or missing PIN"})
                    return
                try:
                    result = outer.app.enqueue_remote_command(
                        str(data.get("command") or ""), data)
                    self._send(200, result)
                except Exception as exc:
                    self._send(500, {"ok": False, "error": str(exc)})

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True, name="remote-control")
        self.thread.start()

    def stop(self):
        if self.httpd is None:
            return
        self.httpd.shutdown()
        self.httpd.server_close()
        self.httpd = None
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
