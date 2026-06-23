"""DearPyGui application shell.

Displays the composite dashboard (built by render.build_dashboard) as a texture
and provides controls: Start/Stop capture, Arm/Stop recording, and treadmill
speed (reusing Treadmill_python/treadmill.py when available). DearPyGui is
imported lazily so the rest of the package runs without it; a lightweight
OpenCV-window backend is provided as a fallback (run_live.py --backend cv2).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

from .render import build_dashboard, dashboard_hit_test, dashboard_resize_hit_test
from ..sources.camera import CameraSource
from ..trials.export import TrialExporter
from ..trials.manager import RecordingDataManager
from ..trials.render import build_trial_review


class DearPyGuiApp:
    def __init__(self, engine, recorder, cfg):
        self.engine = engine
        self.recorder = recorder
        self.cfg = cfg
        self.W = int(cfg.get("gui", "width", default=1600))
        self.H = int(cfg.get("gui", "height", default=900))
        self.window_s = float(cfg.get("gui", "plot_window_s", default=6.0))
        self.body_mass = cfg.get("calibration", "force", "body_mass_kg") or 82.0
        self._buf = np.zeros((self.H, self.W, 4), np.float32)
        self._trial_buf = np.zeros((self.H, self.W, 4), np.float32)
        self.dashboard_focus = None
        self.dashboard_layout_controls = {"left_frac": 0.18, "right_frac": 0.30}
        self._dashboard_drag_splitter = None
        self._last_live_size = (self.W, self.H)
        self._dash_texture_tag = "dash_tex"
        self._dash_texture_counter = 0
        self.preview_w = 320
        self.preview_h = 180
        self._preview_bufs = [
            np.zeros((self.preview_h, self.preview_w, 4), np.float32)
            for _ in range(4)
        ]
        self.preview_cams = {}
        self.preview_sources = {}
        self.calibration_proc = None
        self.calibration_status = self._calibration_summary()
        self.trial_manager = RecordingDataManager(
            cfg.abspath(cfg.get("recording", "output_dir", default="recordings")))
        self.trials = []
        self.selected_trial = None
        self.trial_frame = 0
        self.trial_playing = False
        self._last_trial_tick = 0.0

    def _frame_to_rgba(self, bgr):
        h, w = bgr.shape[:2]
        if self._buf.shape[:2] != (h, w):
            self._buf = np.zeros((h, w, 4), np.float32)
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        self._buf[:, :, :3] = rgb
        self._buf[:, :, 3] = 1.0
        return self._buf.ravel()

    def _image_to_rgba(self, bgr, buf):
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        buf[:, :, :3] = rgb
        buf[:, :, 3] = 1.0
        return buf.ravel()

    def _preview_to_rgba(self, bgr, slot):
        import cv2
        canvas = np.full((self.preview_h, self.preview_w, 3), 18, np.uint8)
        if bgr is not None:
            h, w = bgr.shape[:2]
            scale = min(self.preview_w / max(1, w), self.preview_h / max(1, h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            small = cv2.resize(bgr, (nw, nh))
            x0 = (self.preview_w - nw) // 2
            y0 = (self.preview_h - nh) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = small
        return self._image_to_rgba(canvas, self._preview_bufs[slot])

    def run(self):
        import dearpygui.dearpygui as dpg
        dpg.create_context()

        with dpg.texture_registry(tag="texture_registry"):
            tex = dpg.add_raw_texture(self.W, self.H, self._buf.ravel(),
                                      format=dpg.mvFormat_Float_rgba,
                                      tag=self._dash_texture_tag)
            trial_tex = dpg.add_raw_texture(self.W, self.H, self._trial_buf.ravel(),
                                            format=dpg.mvFormat_Float_rgba,
                                            tag="trial_tex")
            for i, buf in enumerate(self._preview_bufs):
                dpg.add_raw_texture(self.preview_w, self.preview_h, buf.ravel(),
                                    format=dpg.mvFormat_Float_rgba,
                                    tag=f"cam_preview_tex_{i}")

        with dpg.theme(tag="theme_stop"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (180, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 70, 70))
        with dpg.theme(tag="theme_sp_on"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 150, 60))

        with dpg.window(tag="main", label="Live Capture", no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Capture", callback=self._start)
                dpg.add_button(label="Stop Capture", callback=self._stop)
                dpg.add_button(label="Arm Recording", callback=self._arm)
                dpg.add_button(label="Stop+Save", callback=self._save)
                dpg.add_text("idle", tag="rec_status")
            # --- treadmill controls (mirror the existing GUI) ---
            with dpg.group(horizontal=True):
                dpg.add_text("Treadmill")
                dpg.add_text("V [m/s]")
                dpg.add_input_float(tag="inpt_vel", default_value=0.0, width=90,
                                    step=0.1, format="%.2f")
                dpg.add_text("A [m/s2]")
                dpg.add_input_float(tag="inpt_acc", default_value=0.5, width=90,
                                    step=0.1, format="%.2f")
                dpg.add_text("Incl [deg]")
                dpg.add_input_float(tag="inpt_incl", default_value=0.0, width=90,
                                    step=0.5, format="%.1f")
                dpg.add_button(label="SET", callback=self._set_fixed, width=70)
                dpg.add_button(label="STOP", tag="btn_stop", callback=self._tm_stop,
                               width=80)
                dpg.bind_item_theme("btn_stop", "theme_stop")
                dpg.add_button(label="Self-Paced: OFF", tag="btn_sp",
                               callback=self._toggle_self_paced, width=150)
                dpg.add_text("V: 0.00 m/s", tag="rdt_velocity")
            with dpg.tab_bar():
                with dpg.tab(label="Live"):
                    with dpg.child_window(tag="dash_host", width=-1, height=-1,
                                          border=False, no_scrollbar=True):
                        dpg.add_image(tex, tag="dash_image", width=self.W, height=self.H)
                with dpg.tab(label="Cameras"):
                    self._build_cameras_page(dpg)
                with dpg.tab(label="Calibration"):
                    self._build_calibration_page(dpg)
                with dpg.tab(label="Trials"):
                    self._build_trials_page(dpg, trial_tex)

        dpg.create_viewport(title="Treadmill Live MoCap",
                            width=self.W + 30, height=self.H + 90)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

        self._start()
        while dpg.is_dearpygui_running():
            dash_w, dash_h = self._dashboard_size(dpg)
            self._ensure_dashboard_texture(dpg, dash_w, dash_h)
            self._handle_dashboard_resize_drag(dpg, dash_w, dash_h)
            self._handle_dashboard_double_click(dpg, dash_w, dash_h)
            img = build_dashboard(self.engine.state, size=(dash_w, dash_h),
                                  window_s=self.window_s,
                                  body_mass_kg=self.body_mass,
                                  focus=self.dashboard_focus,
                                  layout_controls=self.dashboard_layout_controls)
            dpg.set_value(self._dash_texture_tag, self._frame_to_rgba(img))
            dpg.set_value("rec_status",
                          "RECORDING" if self.recorder.armed else "idle")
            st = self.engine.state.get_status()
            dpg.set_value("rdt_velocity", f"V: {st.get('treadmill_current_vel',0):.2f}"
                          f" m/s ({st.get('treadmill_mode','')})")
            self._update_camera_textures(dpg)
            self._update_calibration_status(dpg)
            self._update_trial_texture(dpg)
            dpg.render_dearpygui_frame()

        self._stop()
        self._close_all_previews()
        dpg.destroy_context()

    def _dashboard_size(self, dpg):
        if not dpg.does_item_exist("dash_host"):
            return self._last_live_size
        try:
            w, h = dpg.get_item_rect_size("dash_host")
        except Exception:
            return self._last_live_size
        w = max(900, int(w or self.W))
        h = max(520, int(h or self.H))
        return w, h

    def _ensure_dashboard_texture(self, dpg, w, h):
        if self._last_live_size == (w, h) and dpg.does_item_exist(self._dash_texture_tag):
            return
        self._last_live_size = (w, h)
        self._buf = np.zeros((h, w, 4), np.float32)
        old_tag = self._dash_texture_tag
        self._dash_texture_counter += 1
        self._dash_texture_tag = f"dash_tex_{self._dash_texture_counter}"
        dpg.add_raw_texture(w, h, self._buf.ravel(),
                            format=dpg.mvFormat_Float_rgba,
                            tag=self._dash_texture_tag,
                            parent="texture_registry")
        if dpg.does_item_exist("dash_image"):
            dpg.configure_item("dash_image", texture_tag=self._dash_texture_tag,
                               width=w, height=h)
        if dpg.does_item_exist(old_tag):
            try:
                dpg.delete_item(old_tag)
            except Exception:
                pass

    def _handle_dashboard_double_click(self, dpg, w, h):
        if not dpg.does_item_exist("dash_image"):
            return
        try:
            hovered = dpg.is_item_hovered("dash_image")
            clicked = dpg.is_mouse_button_double_clicked(button=dpg.mvMouseButton_Left)
        except Exception:
            return
        if not (hovered and clicked):
            return
        mx, my = dpg.get_mouse_pos(local=False)
        ix, iy = dpg.get_item_rect_min("dash_image")
        local = (mx - ix, my - iy)
        if dashboard_resize_hit_test((w, h), local, self.dashboard_layout_controls):
            return
        if self.dashboard_focus:
            self.dashboard_focus = None
            return
        self.dashboard_focus = dashboard_hit_test((w, h), local,
                                                  self.dashboard_layout_controls)

    def _handle_dashboard_resize_drag(self, dpg, w, h):
        if not dpg.does_item_exist("dash_image") or self.dashboard_focus:
            self._dashboard_drag_splitter = None
            return
        try:
            hovered = dpg.is_item_hovered("dash_image")
            down = dpg.is_mouse_button_down(button=dpg.mvMouseButton_Left)
            clicked = dpg.is_mouse_button_clicked(button=dpg.mvMouseButton_Left)
            released = dpg.is_mouse_button_released(button=dpg.mvMouseButton_Left)
        except Exception:
            return
        if released:
            self._dashboard_drag_splitter = None
            return
        mx, my = dpg.get_mouse_pos(local=False)
        ix, iy = dpg.get_item_rect_min("dash_image")
        lx, ly = mx - ix, my - iy
        if clicked and hovered:
            self._dashboard_drag_splitter = dashboard_resize_hit_test(
                (w, h), (lx, ly), self.dashboard_layout_controls)
        if self._dashboard_drag_splitter and down:
            if self._dashboard_drag_splitter == "left":
                left_w = max(140.0, lx - 12.0)
                self.dashboard_layout_controls["left_frac"] = float(
                    np.clip(left_w / max(1, w), 0.12, 0.36))
            elif self._dashboard_drag_splitter == "right":
                right_w = max(220.0, w - lx - 12.0)
                self.dashboard_layout_controls["right_frac"] = float(
                    np.clip(right_w / max(1, w), 0.22, 0.42))

    # callbacks
    def _start(self, *_):
        self.engine.start()

    def _stop(self, *_):
        self.engine.stop()

    def _arm(self, *_):
        self.recorder.arm()

    def _save(self, *_):
        out = self.recorder.stop_and_save()
        print("saved recording ->", out)
        self._refresh_trials()

    # cameras page
    def _build_cameras_page(self, dpg):
        live_cfg = self.cfg.get("cameras", "live", default={}) or {}
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="camera_preview_source",
                               hint="camera index, RTSP/SRT URL, or file path",
                               width=420)
            dpg.add_button(label="Open Preview", callback=self._open_preview_from_ui)
            dpg.add_button(label="Close Previews", callback=lambda *_: self._close_all_previews())
            dpg.add_text("", tag="camera_preview_message")
        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Load cam0 Source",
                           callback=lambda *_: self._set_preview_source(live_cfg.get("cam0", "")),
                           width=150)
            dpg.add_button(label="Load cam1 Source",
                           callback=lambda *_: self._set_preview_source(live_cfg.get("cam1", "")),
                           width=150)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            for i in range(4):
                with dpg.child_window(width=self.preview_w + 18, height=self.preview_h + 86,
                                      border=True):
                    label = "Active cam0" if i == 0 else ("Active cam1" if i == 1 else f"Preview {i - 1}")
                    dpg.add_text(label, tag=f"cam_preview_title_{i}")
                    dpg.add_image(f"cam_preview_tex_{i}")
                    dpg.add_text("idle", tag=f"cam_preview_status_{i}")
                    if i >= 2:
                        dpg.add_button(label="Close", callback=lambda s, a, u=i: self._close_preview(u),
                                       width=80)

    def _set_preview_source(self, source):
        import dearpygui.dearpygui as dpg
        if dpg.does_item_exist("camera_preview_source"):
            dpg.set_value("camera_preview_source", str(source or ""))

    def _open_preview_from_ui(self, *_):
        import dearpygui.dearpygui as dpg
        source = dpg.get_value("camera_preview_source") if dpg.does_item_exist("camera_preview_source") else ""
        self._open_preview_source(source)

    def _open_preview_source(self, source):
        import dearpygui.dearpygui as dpg
        source = str(source or "").strip()
        if not source:
            dpg.set_value("camera_preview_message", "Enter a source first.")
            return
        slot = next((i for i in range(2, 4) if i not in self.preview_cams), None)
        if slot is None:
            dpg.set_value("camera_preview_message", "Close a preview slot first.")
            return
        live_cfg = self.cfg.get("cameras", "live", default={}) or {}
        cam = CameraSource(100 + slot, source,
                           buffersize=live_cfg.get("buffersize", 1),
                           ffmpeg_opts=live_cfg.get("ffmpeg_opts", ""),
                           transport_latency_s=float(self.cfg.get(
                               "cameras", "transport_latency_s", default=0.0) or 0.0),
                           width=live_cfg.get("width"),
                           height=live_cfg.get("height"),
                           fps=live_cfg.get("fps"))
        cam.start()
        self.preview_cams[slot] = cam
        self.preview_sources[slot] = source
        dpg.set_value("camera_preview_message", f"Opened preview {slot - 1}: {source}")

    def _close_preview(self, slot):
        cam = self.preview_cams.pop(slot, None)
        self.preview_sources.pop(slot, None)
        if cam is not None:
            cam.stop()

    def _close_all_previews(self):
        for slot in list(self.preview_cams):
            self._close_preview(slot)

    def _update_camera_textures(self, dpg):
        for slot in range(4):
            frame = None
            status = "idle"
            if slot in (0, 1):
                s, _ = self.engine.state.frame[slot].get()
                frame = s.value if s is not None else None
                cam = self.engine.cams.get(slot)
                if cam is not None:
                    status = f"{cam.fps_est:.1f} fps  {cam.width}x{cam.height}"
            else:
                cam = self.preview_cams.get(slot)
                if cam is not None:
                    s, _ = cam.read_latest()
                    frame = s.value if s is not None else None
                    status = f"{cam.fps_est:.1f} fps  {cam.width}x{cam.height}  {self.preview_sources.get(slot, '')}"
            if dpg.does_item_exist(f"cam_preview_tex_{slot}"):
                dpg.set_value(f"cam_preview_tex_{slot}", self._preview_to_rgba(frame, slot))
            if dpg.does_item_exist(f"cam_preview_status_{slot}"):
                dpg.set_value(f"cam_preview_status_{slot}", status)

    # calibration page
    def _build_calibration_page(self, dpg):
        cb = self.cfg.get("calibration", "cameras", "checkerboard", default={}) or {}
        dpg.add_text("Stereo camera calibration", tag="calib_heading")
        dpg.add_text(self.calibration_status, tag="calib_status")
        dpg.add_text(
            f"Checkerboard: {cb.get('cols')}x{cb.get('rows')} inner corners, "
            f"{float(cb.get('square_m', 0.0) or 0.0) * 1000:.0f} mm squares, "
            f"minimum {cb.get('min_pairs', 12)} pairs",
            tag="calib_checkerboard")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Run Stereo Calibration", callback=self._run_calibration)
            dpg.add_button(label="Stop Calibration", callback=self._stop_calibration)
            dpg.add_button(label="Reload Calibration", callback=self._reload_calibration)
        dpg.add_text("", tag="calib_process_status")
        dpg.add_text("", tag="calib_log_paths")

    def _calibration_summary(self):
        path = self.cfg.abspath(self.cfg.get("calibration", "cameras", "extrinsics"))
        if not path or not os.path.exists(path):
            return "No stereo calibration file found."
        try:
            d = np.load(path)
            parts = [f"Loaded {path}"]
            if "stereo_rms_px" in d:
                parts.append(f"RMS {float(d['stereo_rms_px']):.3f} px")
            if "T" in d:
                parts.append(f"baseline {float(np.linalg.norm(d['T'])):.3f} m")
            if "captured_pairs" in d:
                parts.append(f"pairs {int(d['captured_pairs'])}")
            return " | ".join(parts)
        except Exception as exc:
            return f"Calibration file exists but could not be read: {exc}"

    def _run_calibration(self, *_):
        import dearpygui.dearpygui as dpg
        if self.calibration_proc and self.calibration_proc.poll() is None:
            dpg.set_value("calib_process_status", "Calibration is already running.")
            return
        if self.recorder.armed:
            dpg.set_value("calib_process_status", "Stop/save the recording before calibration.")
            return
        self.engine.stop()
        calib_dir = self.cfg.abspath("calib")
        os.makedirs(calib_dir, exist_ok=True)
        out_log = os.path.join(calib_dir, "calibration_stdout.log")
        err_log = os.path.join(calib_dir, "calibration_stderr.log")
        cmd = [sys.executable, "-m", "livecap.calib.run_calibration",
               "--config", self.cfg.config_path]
        self._calib_out_fh = open(out_log, "w", encoding="utf-8")
        self._calib_err_fh = open(err_log, "w", encoding="utf-8")
        self.calibration_proc = subprocess.Popen(
            cmd, cwd=self.cfg.config_dir,
            stdout=self._calib_out_fh, stderr=self._calib_err_fh)
        dpg.set_value("calib_process_status", "Calibration running in OpenCV window.")
        dpg.set_value("calib_log_paths", f"Logs: {out_log} | {err_log}")

    def _stop_calibration(self, *_):
        import dearpygui.dearpygui as dpg
        if self.calibration_proc and self.calibration_proc.poll() is None:
            self.calibration_proc.terminate()
            dpg.set_value("calib_process_status", "Stopping calibration...")

    def _reload_calibration(self, *_):
        import dearpygui.dearpygui as dpg
        self.engine.reload_calibration()
        self.calibration_status = self._calibration_summary()
        if dpg.does_item_exist("calib_status"):
            dpg.set_value("calib_status", self.calibration_status)
        dpg.set_value("calib_process_status", "Calibration reloaded.")

    def _update_calibration_status(self, dpg):
        if self.calibration_proc and self.calibration_proc.poll() is not None:
            code = self.calibration_proc.returncode
            for name in ("_calib_out_fh", "_calib_err_fh"):
                fh = getattr(self, name, None)
                if fh is not None:
                    fh.close()
                    setattr(self, name, None)
            self.calibration_proc = None
            self.engine.reload_calibration()
            self.calibration_status = self._calibration_summary()
            if dpg.does_item_exist("calib_status"):
                dpg.set_value("calib_status", self.calibration_status)
            if dpg.does_item_exist("calib_process_status"):
                dpg.set_value("calib_process_status",
                              "Calibration complete." if code == 0
                              else f"Calibration exited with code {code}.")

    # trials page
    def _build_trials_page(self, dpg, trial_tex):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=520, height=self.H - 120, tag="trial_browser"):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="trial_search", hint="Search trials", width=300,
                                       callback=lambda *_: self._refresh_trials())
                    dpg.add_button(label="Refresh", callback=lambda *_: self._refresh_trials())
                dpg.add_spacer(height=6)
                dpg.add_child_window(tag="trial_table_host", height=360, border=False)
                dpg.add_separator()
                dpg.add_text("Steps")
                dpg.add_child_window(tag="step_table_host", height=300, border=False)
            with dpg.child_window(width=-1, height=self.H - 120, tag="trial_detail"):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play", tag="trial_play_btn", callback=self._toggle_trial_play)
                    dpg.add_button(label="<", callback=lambda *_: self._nudge_trial_frame(-1))
                    dpg.add_button(label=">", callback=lambda *_: self._nudge_trial_frame(1))
                    dpg.add_slider_int(tag="trial_frame_slider", min_value=0, max_value=0,
                                       default_value=0, width=360, callback=self._set_trial_frame)
                    dpg.add_combo(["both", "raw", "processed"], tag="export_scope",
                                  default_value="both", width=120)
                    dpg.add_button(label="JSON", callback=lambda *_: self._export_trial("json"))
                    dpg.add_button(label="CSV", callback=lambda *_: self._export_trial("csv"))
                    dpg.add_button(label="Excel", callback=lambda *_: self._export_trial("excel"))
                    dpg.add_button(label="Archive", callback=lambda *_: self._export_trial("archive"))
                dpg.add_text("Select a trial", tag="trial_status")
                dpg.add_image(trial_tex)
        self._refresh_trials()

    def _refresh_trials(self):
        import dearpygui.dearpygui as dpg
        query = dpg.get_value("trial_search") if dpg.does_item_exist("trial_search") else ""
        self.trials = self.trial_manager.search(query)
        if not dpg.does_item_exist("trial_table_host"):
            return
        dpg.delete_item("trial_table_host", children_only=True)
        with dpg.table(parent="trial_table_host", header_row=True, resizable=True,
                       borders_innerH=True, borders_innerV=False, borders_outerH=True):
            for name in ("Trial name", "Date", "Duration", "Steps", "Distance", "Speed", "Cadence"):
                dpg.add_table_column(label=name)
            for item in self.trials:
                with dpg.table_row():
                    label = item.get("recording_name") or item.get("path")
                    dpg.add_button(label=str(label), width=150,
                                   callback=lambda s, a, u=item: self._select_trial(u))
                    dpg.add_text(str(item.get("date_time") or "")[:19])
                    dpg.add_text(self._fmt_ui(item.get("duration_s"), "s"))
                    dpg.add_text(str(item.get("total_steps") or 0))
                    dpg.add_text(self._fmt_ui(item.get("distance_m"), "m"))
                    dpg.add_text(self._fmt_ui(item.get("average_speed_m_s"), "m/s"))
                    dpg.add_text(self._fmt_ui(item.get("average_cadence_steps_per_min"), "/min", 0))

    @staticmethod
    def _fmt_ui(value, suffix="", digits=1):
        try:
            v = float(value)
            if not np.isfinite(v):
                return "--"
            return f"{v:.{digits}f} {suffix}".strip()
        except Exception:
            return "--"

    def _select_trial(self, item):
        import dearpygui.dearpygui as dpg
        self.selected_trial = self.trial_manager.get_trial(item.get("trial_id") or item.get("path"))
        self.trial_frame = 0
        self.trial_playing = False
        dpg.set_item_label("trial_play_btn", "Play")
        n = max(0, (self.selected_trial.frame_count() - 1) if self.selected_trial else 0)
        dpg.configure_item("trial_frame_slider", max_value=n)
        dpg.set_value("trial_frame_slider", 0)
        if self.selected_trial:
            meta = self.selected_trial.metadata
            summ = self.selected_trial.summary
            dpg.set_value(
                "trial_status",
                f"{meta.get('recording_name', self.selected_trial.path.name)} | "
                f"{summ.get('total_steps', 0)} steps | "
                f"{self._fmt_ui(summ.get('recording_duration_s'), 's')}",
            )
        self._rebuild_step_table()

    def _rebuild_step_table(self):
        import dearpygui.dearpygui as dpg
        if not dpg.does_item_exist("step_table_host"):
            return
        dpg.delete_item("step_table_host", children_only=True)
        if not self.selected_trial:
            return
        with dpg.table(parent="step_table_host", header_row=True, resizable=True,
                       borders_innerH=True, borders_outerH=True):
            for name in ("Step", "Foot", "Time", "Stride", "Stance"):
                dpg.add_table_column(label=name)
            for step in self.selected_trial.steps:
                with dpg.table_row():
                    dpg.add_button(label=step.get("step_id", ""),
                                   callback=lambda s, a, u=step: self._jump_to_step(u))
                    dpg.add_text(step.get("foot", ""))
                    dpg.add_text(self._fmt_ui(step.get("timestamp"), "s"))
                    dpg.add_text(self._fmt_ui(
                        step.get("spatial_metrics", {}).get("stride_length_m"), "m", 2))
                    dpg.add_text(self._fmt_ui(
                        step.get("temporal_metrics", {}).get("stance_time_s"), "s", 2))

    def _jump_to_step(self, step):
        import dearpygui.dearpygui as dpg
        self.trial_frame = int(step.get("start_frame") or 0)
        if dpg.does_item_exist("trial_frame_slider"):
            dpg.set_value("trial_frame_slider", self.trial_frame)

    def _set_trial_frame(self, sender, app_data):
        self.trial_frame = int(app_data or 0)

    def _nudge_trial_frame(self, delta):
        import dearpygui.dearpygui as dpg
        max_frame = dpg.get_item_configuration("trial_frame_slider").get("max_value", 0)
        self.trial_frame = int(np.clip(self.trial_frame + delta, 0, max_frame))
        dpg.set_value("trial_frame_slider", self.trial_frame)

    def _toggle_trial_play(self, *_):
        import dearpygui.dearpygui as dpg
        self.trial_playing = not self.trial_playing
        self._last_trial_tick = time.time()
        dpg.set_item_label("trial_play_btn", "Pause" if self.trial_playing else "Play")

    def _export_trial(self, fmt):
        import dearpygui.dearpygui as dpg
        if not self.selected_trial:
            return
        scope = dpg.get_value("export_scope") if dpg.does_item_exist("export_scope") else "both"
        try:
            out = TrialExporter(self.selected_trial).export(fmt, scope)
            dpg.set_value("trial_status", f"Exported {out}")
            print("exported trial ->", out)
        except Exception as exc:
            dpg.set_value("trial_status", f"Export failed: {exc}")
            print("[trials] export failed:", exc)

    def _update_trial_texture(self, dpg):
        if self.trial_playing and self.selected_trial:
            now = time.time()
            fps = float(self.cfg.get("pose", "target_fps", default=30) or 30)
            if now - self._last_trial_tick >= 1.0 / fps:
                max_frame = max(0, self.selected_trial.frame_count() - 1)
                self.trial_frame = 0 if self.trial_frame >= max_frame else self.trial_frame + 1
                if dpg.does_item_exist("trial_frame_slider"):
                    dpg.set_value("trial_frame_slider", self.trial_frame)
                self._last_trial_tick = now
        img = build_trial_review(self.selected_trial, self.trial_frame,
                                 size=(self.W, self.H))
        dpg.set_value("trial_tex", self._image_to_rgba(img, self._trial_buf))

    # treadmill controls
    def _set_fixed(self, *_):
        import dearpygui.dearpygui as dpg
        self.engine.treadmill_ctrl.set_incline(dpg.get_value("inpt_incl"))
        self.engine.treadmill_ctrl.set_fixed(dpg.get_value("inpt_vel"),
                                             dpg.get_value("inpt_acc"))
        dpg.set_item_label("btn_sp", "Self-Paced: OFF")
        dpg.bind_item_theme("btn_sp", 0)

    def _tm_stop(self, *_):
        import dearpygui.dearpygui as dpg
        self.engine.treadmill_ctrl.stop()
        dpg.set_value("inpt_vel", 0.0)
        dpg.set_item_label("btn_sp", "Self-Paced: OFF")
        dpg.bind_item_theme("btn_sp", 0)

    def _toggle_self_paced(self, *_):
        import dearpygui.dearpygui as dpg
        tc = self.engine.treadmill_ctrl
        if tc.self_paced:
            tc.stop_self_paced()
            dpg.set_item_label("btn_sp", "Self-Paced: OFF")
            dpg.bind_item_theme("btn_sp", 0)
        else:
            tc.start_self_paced(initial_vel=dpg.get_value("inpt_vel"))
            dpg.set_item_label("btn_sp", "Self-Paced: ON")
            dpg.bind_item_theme("btn_sp", "theme_sp_on")


def run_cv2_backend(engine, recorder, cfg):
    """Fallback live GUI using an OpenCV window (no DearPyGui dependency).

    Keys:
      r=arm recording   s=stop+save        q=quit
      [ / ] = target speed -/+ 0.1 m/s     g=SET (apply)   x=STOP
      , / . = incline -/+ 0.5 deg          p=toggle Self-Paced
    """
    import cv2
    W = int(cfg.get("gui", "width", default=1600))
    H = int(cfg.get("gui", "height", default=900))
    window_s = float(cfg.get("gui", "plot_window_s", default=6.0))
    body_mass = cfg.get("calibration", "force", "body_mass_kg") or 82.0
    tc = engine.treadmill_ctrl
    target_v, target_a, target_i = 0.0, 0.5, 0.0
    focus = None
    layout_controls = {"left_frac": 0.18, "right_frac": 0.30}
    drag_splitter = None
    engine.start()
    win = "Treadmill Live MoCap"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, *_):
        nonlocal focus, W, H, drag_splitter
        if event == cv2.EVENT_LBUTTONDOWN and not focus:
            drag_splitter = dashboard_resize_hit_test((W, H), (x, y), layout_controls)
            return
        if event == cv2.EVENT_MOUSEMOVE and drag_splitter:
            if drag_splitter == "left":
                layout_controls["left_frac"] = float(np.clip((x - 12.0) / max(1, W), 0.12, 0.36))
            elif drag_splitter == "right":
                layout_controls["right_frac"] = float(np.clip((W - x - 12.0) / max(1, W), 0.22, 0.42))
            return
        if event == cv2.EVENT_LBUTTONUP:
            drag_splitter = None
            return
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return
        if dashboard_resize_hit_test((W, H), (x, y), layout_controls):
            return
        if focus:
            focus = None
        else:
            focus = dashboard_hit_test((W, H), (x, y), layout_controls)

    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            try:
                _, _, ww, wh = cv2.getWindowImageRect(win)
                if ww > 0 and wh > 0:
                    W, H = max(900, int(ww)), max(520, int(wh))
            except Exception:
                pass
            img = build_dashboard(engine.state, size=(W, H),
                                  window_s=window_s, body_mass_kg=body_mass,
                                  focus=focus,
                                  layout_controls=layout_controls)
            cv2.imshow(win, img)
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                recorder.arm()
            elif k == ord("s"):
                print("saved ->", recorder.stop_and_save())
            elif k == ord("]"):
                target_v = min(2.0, target_v + 0.1)
            elif k == ord("["):
                target_v = max(0.0, target_v - 0.1)
            elif k == ord("."):
                target_i += 0.5
                tc.set_incline(target_i)
            elif k == ord(","):
                target_i = max(0.0, target_i - 0.5)
                tc.set_incline(target_i)
            elif k == ord("g"):
                tc.set_incline(target_i)
                tc.set_fixed(target_v, target_a)
            elif k == ord("x"):
                tc.stop()
                target_v = 0.0
            elif k == ord("p"):
                if tc.self_paced:
                    tc.stop_self_paced()
                else:
                    tc.start_self_paced(initial_vel=target_v)
    finally:
        engine.stop()
        cv2.destroyAllWindows()
