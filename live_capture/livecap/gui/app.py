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
import threading
import time
import traceback

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
        self.main_overlay_camera = int(cfg.get("gui", "main_overlay_camera", default=1) or 1)
        self._buf = np.zeros((self.H, self.W, 4), np.float32)
        self._trial_buf = np.zeros((self.H, self.W, 4), np.float32)
        self._trial_texture_size = (self.W, self.H)
        self._trial_texture_tag = "trial_tex"
        self._trial_texture_counter = 0
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
        self.trial_chart = "Overview"
        self.trial_replay_camera = "auto"
        self.trial_play_speed = 1.0
        self._trial_play_start_wall = 0.0
        self._trial_play_start_time = 0.0
        self._trial_dirty = True
        self._last_trial_tick = 0.0
        self._trial_auto_selected = False
        self._trial_dragging_timeline = False
        self._processing_thread = None
        self._processing_lock = threading.Lock()
        self._processing_state = {
            "running": False,
            "done": False,
            "error": None,
            "progress": 0.0,
            "message": "Idle",
            "trial_path": None,
            "result_path": None,
        }
        self._processing_notice_consumed = True

    def _frame_to_rgba(self, bgr):
        h, w = bgr.shape[:2]
        if self._buf.shape[:2] != (h, w):
            self._buf = np.zeros((h, w, 4), np.float32)
            self._buf[:, :, 3] = 1.0
        np.multiply(bgr[:, :, 2], 1.0 / 255.0, out=self._buf[:, :, 0],
                    casting="unsafe")
        np.multiply(bgr[:, :, 1], 1.0 / 255.0, out=self._buf[:, :, 1],
                    casting="unsafe")
        np.multiply(bgr[:, :, 0], 1.0 / 255.0, out=self._buf[:, :, 2],
                    casting="unsafe")
        self._buf[:, :, 3] = 1.0
        return self._buf.ravel()

    def _image_to_rgba(self, bgr, buf):
        np.multiply(bgr[:, :, 2], 1.0 / 255.0, out=buf[:, :, 0],
                    casting="unsafe")
        np.multiply(bgr[:, :, 1], 1.0 / 255.0, out=buf[:, :, 1],
                    casting="unsafe")
        np.multiply(bgr[:, :, 0], 1.0 / 255.0, out=buf[:, :, 2],
                    casting="unsafe")
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
                                            tag=self._trial_texture_tag)
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
                dpg.add_text("Skeleton overlay")
                dpg.add_combo(["cam0", "cam1"], tag="overlay_camera_combo",
                              default_value=f"cam{self.main_overlay_camera}",
                              width=90, callback=self._set_overlay_camera)
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
                                  layout_controls=self.dashboard_layout_controls,
                                  overlay_camera=self.main_overlay_camera)
            dpg.set_value(self._dash_texture_tag, self._frame_to_rgba(img))
            dpg.set_value("rec_status",
                          "RECORDING" if self.recorder.armed else "idle")
            st = self.engine.state.get_status()
            dpg.set_value("rdt_velocity", f"V: {st.get('treadmill_current_vel',0):.2f}"
                          f" m/s ({st.get('treadmill_mode','')})")
            self._update_camera_textures(dpg)
            self._update_calibration_status(dpg)
            self._update_processing_ui(dpg)
            self._ensure_trial_texture(dpg)
            self._handle_trial_image_input(dpg)
            self._update_trial_texture(dpg)
            dpg.render_dearpygui_frame()

        self._stop()
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join()
        self._process_pending_trials(show_status=False)
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
    def _set_overlay_camera(self, sender=None, app_data=None, *_):
        value = str(app_data if app_data is not None else "").strip().lower()
        self.main_overlay_camera = 1 if value == "cam1" else 0

    def _start(self, *_):
        self.engine.start()

    def _stop(self, *_):
        self.engine.stop()

    def _arm(self, *_):
        self.recorder.arm()

    def _save(self, *_):
        out = self.recorder.stop_and_save()
        print("saved recording ->", out)
        try:
            import dearpygui.dearpygui as dpg
            if dpg.does_item_exist("rec_status"):
                dpg.set_value("rec_status", "saved raw; processing queued")
        except Exception:
            pass
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
                dpg.add_text(str(self.trial_manager.recordings_dir), tag="trial_root_text",
                             wrap=500)
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="trial_search", hint="Search trials", width=300,
                                       callback=lambda *_: self._refresh_trials())
                    dpg.add_button(label="Refresh", callback=lambda *_: self._refresh_trials())
                    dpg.add_button(label="Load Latest", callback=self._load_latest_trial)
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
                    dpg.add_text("0.00s / 0.00s", tag="trial_time_text")
                    dpg.add_combo(["0.25x", "0.5x", "1x", "2x"], tag="trial_speed_combo",
                                  default_value="1x", width=75,
                                  callback=self._set_trial_play_speed)
                    dpg.add_combo(self._chart_options(), tag="trial_chart_combo",
                                  default_value="Overview", width=190,
                                  callback=self._set_trial_chart)
                    dpg.add_combo(["auto", "cam0", "cam1"], tag="trial_replay_camera_combo",
                                  default_value="auto", width=90,
                                  callback=self._set_trial_replay_camera)
                    dpg.add_button(label="Prev Chart", callback=lambda *_: self._cycle_trial_chart(-1))
                    dpg.add_button(label="Next Chart", callback=lambda *_: self._cycle_trial_chart(1))
                    dpg.add_button(label="Process", tag="trial_process_btn",
                                   callback=self._process_selected_trial)
                    dpg.add_button(label="Open Overlay", callback=self._open_selected_overlay)
                    dpg.add_combo(["both", "raw", "processed"], tag="export_scope",
                                  default_value="both", width=120)
                    dpg.add_button(label="JSON", callback=lambda *_: self._export_trial("json"))
                    dpg.add_button(label="CSV", callback=lambda *_: self._export_trial("csv"))
                    dpg.add_button(label="Excel", callback=lambda *_: self._export_trial("excel"))
                    dpg.add_button(label="Archive", callback=lambda *_: self._export_trial("archive"))
                dpg.add_text("Select a trial", tag="trial_status")
                with dpg.group(horizontal=True):
                    dpg.add_progress_bar(default_value=0.0, width=360,
                                         tag="trial_processing_bar")
                    dpg.add_text("Idle", tag="trial_processing_text")
                dpg.add_image(trial_tex, tag="trial_image",
                              width=self.W, height=self.H)
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
            for name in ("Trial name", "Status", "Date", "Duration", "Steps", "Distance", "Speed", "Cadence"):
                dpg.add_table_column(label=name)
            for row_idx, item in enumerate(self.trials):
                with dpg.table_row():
                    label = item.get("recording_name") or item.get("path")
                    tag = f"trial_select_{row_idx}"
                    dpg.add_button(label=str(label), tag=tag, width=150,
                                   callback=lambda s, a, u=item: self._select_trial(u))
                    with dpg.popup(tag, mousebutton=dpg.mvMouseButton_Right):
                        dpg.add_button(label="Select",
                                       callback=lambda s, a, u=item: self._select_trial(u))
                        dpg.add_button(label="Process",
                                       callback=lambda s, a, u=item: self._process_trial_item(u))
                    dpg.add_text(str(item.get("status") or "raw_saved"))
                    dpg.add_text(str(item.get("date_time") or "")[:19])
                    dpg.add_text(self._fmt_ui(item.get("duration_s"), "s"))
                    dpg.add_text(str(item.get("total_steps") or 0))
                    dpg.add_text(self._fmt_ui(item.get("distance_m"), "m"))
                    dpg.add_text(self._fmt_ui(item.get("average_speed_m_s"), "m/s"))
                    dpg.add_text(self._fmt_ui(item.get("average_cadence_steps_per_min"), "/min", 0))
        if self.trials and not self.selected_trial and not self._trial_auto_selected:
            self._trial_auto_selected = True
            self._select_trial(self.trials[0])
        elif not self.trials and dpg.does_item_exist("trial_status"):
            dpg.set_value("trial_status", "No trials found in the active recordings folder.")

    def _load_latest_trial(self, *_):
        self._refresh_trials()
        if self.trials:
            self._select_trial(self.trials[0])

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
        self.trial_chart = "Overview"
        self.trial_replay_camera = "auto"
        self._trial_play_start_wall = 0.0
        self._trial_play_start_time = 0.0
        self._trial_dirty = True
        dpg.set_item_label("trial_play_btn", "Play")
        if dpg.does_item_exist("trial_chart_combo"):
            dpg.set_value("trial_chart_combo", self.trial_chart)
        if dpg.does_item_exist("trial_replay_camera_combo"):
            dpg.set_value("trial_replay_camera_combo", self.trial_replay_camera)
        if dpg.does_item_exist("trial_speed_combo"):
            dpg.set_value("trial_speed_combo", f"{self.trial_play_speed:g}x")
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
            self._update_trial_texture(dpg)
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
        self._trial_dirty = True
        if dpg.does_item_exist("trial_frame_slider"):
            dpg.set_value("trial_frame_slider", self.trial_frame)

    def _set_trial_frame(self, sender, app_data):
        self.trial_frame = int(app_data or 0)
        self._reset_trial_play_anchor()
        self._trial_dirty = True

    def _nudge_trial_frame(self, delta):
        import dearpygui.dearpygui as dpg
        max_frame = dpg.get_item_configuration("trial_frame_slider").get("max_value", 0)
        self.trial_frame = int(np.clip(self.trial_frame + delta, 0, max_frame))
        self._reset_trial_play_anchor()
        self._trial_dirty = True
        dpg.set_value("trial_frame_slider", self.trial_frame)

    def _toggle_trial_play(self, *_):
        import dearpygui.dearpygui as dpg
        self.trial_playing = not self.trial_playing
        self._reset_trial_play_anchor()
        self._last_trial_tick = time.time()
        dpg.set_item_label("trial_play_btn", "Pause" if self.trial_playing else "Play")

    def _set_trial_play_speed(self, sender=None, app_data=None, *_):
        text = str(app_data or "1x").strip().lower().replace("x", "")
        try:
            self.trial_play_speed = max(0.05, float(text))
        except Exception:
            self.trial_play_speed = 1.0
        self._reset_trial_play_anchor()

    def _trial_timestamps(self):
        if not self.selected_trial:
            return np.zeros(0)
        frames = self.selected_trial.frames
        return np.asarray(frames.get("timestamp", []), float)

    def _trial_duration(self):
        if not self.selected_trial:
            return 0.0
        try:
            duration = float(self.selected_trial.summary.get("recording_duration_s") or 0.0)
        except Exception:
            duration = 0.0
        ts = self._trial_timestamps()
        if duration <= 0 and ts.size:
            duration = float(np.nanmax(ts))
        return max(0.0, duration)

    def _trial_time_for_frame(self, frame_idx=None):
        ts = self._trial_timestamps()
        if ts.size == 0:
            return 0.0
        idx = int(np.clip(self.trial_frame if frame_idx is None else frame_idx, 0, ts.size - 1))
        return float(ts[idx])

    def _frame_for_trial_time(self, target_t):
        ts = self._trial_timestamps()
        if ts.size == 0:
            return 0
        idx = int(np.clip(np.searchsorted(ts, target_t, side="left"), 0, ts.size - 1))
        if idx > 0 and abs(ts[idx - 1] - target_t) <= abs(ts[idx] - target_t):
            idx -= 1
        return idx

    def _reset_trial_play_anchor(self):
        self._trial_play_start_wall = time.time()
        self._trial_play_start_time = self._trial_time_for_frame()

    @staticmethod
    def _chart_options():
        return [
            "Overview",
            "Camera skeleton replay",
            "Walking speed vs time",
            "Cadence vs time",
            "Stride length vs step",
            "Step length vs step",
            "Joint angles over time",
            "Center of mass trajectory",
            "Foot trajectories",
            "Pressure over time",
            "Ground contact timeline",
            "Left vs right symmetry",
        ]

    def _set_trial_chart(self, sender=None, app_data=None, *_):
        self.trial_chart = str(app_data or "Overview")
        self._trial_dirty = True

    def _set_trial_replay_camera(self, sender=None, app_data=None, *_):
        self.trial_replay_camera = str(app_data or "auto")
        self._trial_dirty = True

    def _cycle_trial_chart(self, delta):
        import dearpygui.dearpygui as dpg
        opts = self._chart_options()
        try:
            i = opts.index(self.trial_chart)
        except ValueError:
            i = 0
        self.trial_chart = opts[(i + int(delta)) % len(opts)]
        if dpg.does_item_exist("trial_chart_combo"):
            dpg.set_value("trial_chart_combo", self.trial_chart)
        self._trial_dirty = True

    def _process_selected_trial(self, *_):
        if self.selected_trial:
            self._process_trial(self.selected_trial)

    def _open_selected_overlay(self, *_):
        import dearpygui.dearpygui as dpg
        if not self.selected_trial:
            return
        rel = self.selected_trial.metadata.get("paths", {}).get("playback_video")
        path = self.selected_trial.path / rel if rel else self.selected_trial.path / "playback" / "skeleton_overlay.mp4"
        if not path.exists():
            if dpg.does_item_exist("trial_status"):
                dpg.set_value("trial_status", "No processed overlay video found for this trial.")
            return
        try:
            os.startfile(path)
        except Exception as exc:
            if dpg.does_item_exist("trial_status"):
                dpg.set_value("trial_status", f"Could not open overlay: {exc}")

    def _process_trial_item(self, item):
        trial = self.trial_manager.get_trial(item.get("trial_id") or item.get("path"))
        if trial:
            self._process_trial(trial)

    def _process_trial(self, trial):
        import dearpygui.dearpygui as dpg
        if self._is_processing():
            if dpg.does_item_exist("trial_status"):
                dpg.set_value("trial_status", "Processing is already running.")
            return
        label = trial.metadata.get("recording_name", trial.path.name)
        if dpg.does_item_exist("trial_status"):
            dpg.set_value("trial_status", f"Processing {label}...")
        self._set_processing_state(
            running=True, done=False, error=None, progress=0.02,
            message=f"Queued {label}", trial_path=str(trial.path),
            result_path=None)
        self._processing_notice_consumed = False
        if dpg.does_item_exist("trial_process_btn"):
            dpg.configure_item("trial_process_btn", enabled=False)
        self._processing_thread = threading.Thread(
            target=self._process_trial_worker, args=(trial.path,),
            daemon=True, name="trial-processing")
        self._processing_thread.start()

    def _process_trial_worker(self, trial_path):
        print(f"[trials] processing -> {trial_path}")
        log_path = trial_path / "processing.log"
        try:
            def progress(stage, current=0, total=1, message=""):
                frac = self._processing_fraction(stage, current, total)
                self._set_processing_state(
                    running=True, done=False, error=None, progress=frac,
                    message=message or stage, trial_path=str(trial_path))
                try:
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                                 f"{stage} {current}/{total} {message}\n")
                except Exception:
                    pass

            processed = self.trial_manager.process_trial(
                str(trial_path.name), self.cfg, progress=progress)
            self._set_processing_state(
                running=False, done=True, error=None, progress=1.0,
                message="Processing complete",
                trial_path=str(trial_path),
                result_path=str(processed.path) if processed else None)
        except Exception as exc:
            tb = traceback.format_exc()
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(tb)
            except Exception:
                pass
            print("[trials] processing failed:", exc)
            print(tb)
            self._set_processing_state(
                running=False, done=True, error=str(exc), progress=0.0,
                message=f"Processing failed: {exc}",
                trial_path=str(trial_path), result_path=None)

    @staticmethod
    def _processing_fraction(stage, current, total):
        total = max(1.0, float(total or 1))
        current = float(current or 0)
        within = float(np.clip(current / total, 0.0, 1.0))
        ranges = {
            "start": (0.00, 0.03),
            "load_model": (0.03, 0.12),
            "precision_pose": (0.12, 0.78),
            "artifacts": (0.78, 0.88),
            "playback": (0.88, 0.98),
            "complete": (0.98, 1.00),
        }
        lo, hi = ranges.get(stage, (0.02, 0.98))
        return float(lo + (hi - lo) * within)

    def _set_processing_state(self, **updates):
        with self._processing_lock:
            self._processing_state.update(updates)

    def _get_processing_state(self):
        with self._processing_lock:
            return dict(self._processing_state)

    def _is_processing(self):
        return bool(self._get_processing_state().get("running"))

    def _update_processing_ui(self, dpg):
        state = self._get_processing_state()
        if dpg.does_item_exist("trial_processing_bar"):
            dpg.set_value("trial_processing_bar",
                          float(np.clip(state.get("progress", 0.0), 0.0, 1.0)))
        if dpg.does_item_exist("trial_processing_text"):
            pct = int(round(float(state.get("progress", 0.0)) * 100))
            msg = state.get("message") or "Idle"
            dpg.set_value("trial_processing_text", f"{pct}%  {msg}")
        if dpg.does_item_exist("trial_process_btn"):
            dpg.configure_item("trial_process_btn", enabled=not state.get("running"))
        if state.get("done") and not self._processing_notice_consumed:
            self._processing_notice_consumed = True
            if state.get("error"):
                if dpg.does_item_exist("trial_status"):
                    dpg.set_value("trial_status", state.get("message"))
                return
            result_path = state.get("result_path")
            if result_path:
                self.selected_trial = self.trial_manager.get_trial(os.path.basename(result_path))
                self.trial_frame = 0
                self.trial_playing = False
                self._trial_dirty = True
                self._refresh_trials()
                self._rebuild_step_table()
                if self.selected_trial and dpg.does_item_exist("trial_frame_slider"):
                    dpg.configure_item("trial_frame_slider",
                                       max_value=max(0, self.selected_trial.frame_count() - 1))
                    dpg.set_value("trial_frame_slider", 0)
                if self.selected_trial and dpg.does_item_exist("trial_status"):
                    path = self.selected_trial.metadata.get("paths", {}).get("playback_video")
                    suffix = f" | playback: {path}" if path else ""
                    dpg.set_value("trial_status", f"Processing complete{suffix}")

    def _process_pending_trials(self, show_status=True):
        try:
            pending = self.trial_manager.pending_trials()
        except Exception as exc:
            print("[trials] could not list pending trials:", exc)
            return
        if not pending:
            return
        if show_status:
            import dearpygui.dearpygui as dpg
            if dpg.does_item_exist("trial_status"):
                dpg.set_value("trial_status", f"Processing {len(pending)} pending trial(s)...")
        print(f"[trials] processing {len(pending)} pending trial(s) before close")
        for trial in pending:
            try:
                self.trial_manager.process_trial(trial, self.cfg)
            except Exception as exc:
                print(f"[trials] pending processing failed for {trial.path}: {exc}")
        self._refresh_trials()

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
        if not self.selected_trial:
            return
        if self.trial_playing and self.selected_trial:
            now = time.time()
            target_t = self._trial_play_start_time + (
                now - self._trial_play_start_wall) * self.trial_play_speed
            duration = self._trial_duration()
            if duration > 0 and target_t > duration:
                target_t = 0.0
                self._trial_play_start_time = 0.0
                self._trial_play_start_wall = now
            new_frame = self._frame_for_trial_time(target_t)
            if new_frame != self.trial_frame or now - self._last_trial_tick >= 0.25:
                self.trial_frame = new_frame
                if dpg.does_item_exist("trial_frame_slider"):
                    dpg.set_value("trial_frame_slider", self.trial_frame)
                self._last_trial_tick = now
                self._trial_dirty = True
        self._update_trial_time_text(dpg)
        if not (self.trial_playing or self._trial_dirty):
            return
        tw, th = self._trial_texture_size
        img = build_trial_review(self.selected_trial, self.trial_frame,
                                 size=(tw, th),
                                 chart_name=self.trial_chart,
                                 replay_camera=self.trial_replay_camera)
        dpg.set_value(self._trial_texture_tag, self._image_to_rgba(img, self._trial_buf))
        self._trial_dirty = False

    def _update_trial_time_text(self, dpg):
        if not dpg.does_item_exist("trial_time_text"):
            return
        cur = self._trial_time_for_frame()
        duration = self._trial_duration()
        dpg.set_value("trial_time_text", f"{cur:.2f}s / {duration:.2f}s")

    def _handle_trial_image_input(self, dpg):
        if not (self.selected_trial and dpg.does_item_exist("trial_image")):
            self._trial_dragging_timeline = False
            return
        try:
            hovered = dpg.is_item_hovered("trial_image")
            clicked = dpg.is_mouse_button_clicked(button=dpg.mvMouseButton_Left)
            down = dpg.is_mouse_button_down(button=dpg.mvMouseButton_Left)
            released = dpg.is_mouse_button_released(button=dpg.mvMouseButton_Left)
        except Exception:
            return
        if released:
            self._trial_dragging_timeline = False
            return
        if not (hovered or self._trial_dragging_timeline):
            return
        mx, my = dpg.get_mouse_pos(local=False)
        ix, iy = dpg.get_item_rect_min("trial_image")
        lx, ly = mx - ix, my - iy
        tw, th = self._trial_texture_size
        timeline = (20, 140, max(1, tw - 40), 86)
        tx, ty, twidth, theight = timeline
        inside_timeline = tx <= lx <= tx + twidth and ty <= ly <= ty + theight
        if clicked and inside_timeline:
            self._trial_dragging_timeline = True
        if self._trial_dragging_timeline and down:
            self._jump_to_timeline_x(lx)

    def _jump_to_timeline_x(self, local_x):
        import dearpygui.dearpygui as dpg
        frames = self.selected_trial.frames if self.selected_trial else {}
        ts = np.asarray(frames.get("timestamp", []), float)
        if ts.size == 0:
            return
        tw, _ = self._trial_texture_size
        px = 36.0
        pw = max(1.0, float(tw - 72))
        frac = float(np.clip((local_x - px) / pw, 0.0, 1.0))
        duration = float(self.selected_trial.summary.get("recording_duration_s") or np.nanmax(ts) or 0.0)
        target_t = frac * max(duration, 0.0)
        idx = int(np.clip(np.searchsorted(ts, target_t, side="left"), 0, ts.size - 1))
        if idx > 0 and abs(ts[idx - 1] - target_t) <= abs(ts[idx] - target_t):
            idx -= 1
        self.trial_frame = idx
        self.trial_playing = False
        self._trial_dirty = True
        if dpg.does_item_exist("trial_frame_slider"):
            dpg.set_value("trial_frame_slider", self.trial_frame)
        if dpg.does_item_exist("trial_play_btn"):
            dpg.set_item_label("trial_play_btn", "Play")

    def _ensure_trial_texture(self, dpg):
        if not dpg.does_item_exist("trial_detail"):
            return
        try:
            w, h = dpg.get_item_rect_size("trial_detail")
        except Exception:
            return
        w = max(700, int(w or self.W))
        h = max(500, int((h or self.H) - 54))
        if self._trial_texture_size == (w, h):
            return
        self._trial_texture_size = (w, h)
        self._trial_buf = np.zeros((h, w, 4), np.float32)
        old_tag = self._trial_texture_tag
        self._trial_texture_counter += 1
        self._trial_texture_tag = f"trial_tex_{self._trial_texture_counter}"
        dpg.add_raw_texture(w, h, self._trial_buf.ravel(),
                            format=dpg.mvFormat_Float_rgba,
                            tag=self._trial_texture_tag,
                            parent="texture_registry")
        if dpg.does_item_exist("trial_image"):
            dpg.configure_item("trial_image", texture_tag=self._trial_texture_tag,
                               width=w, height=h)
        if dpg.does_item_exist(old_tag):
            try:
                dpg.delete_item(old_tag)
            except Exception:
                pass
        self._trial_dirty = True

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
    main_overlay_camera = int(cfg.get("gui", "main_overlay_camera", default=1) or 1)
    tc = engine.treadmill_ctrl
    target_v, target_a, target_i = 0.0, 0.5, 0.0
    focus = None
    layout_controls = {"left_frac": 0.18, "right_frac": 0.30}
    drag_splitter = None
    engine.start()
    win = "Treadmill Live MoCap"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, *_):
        nonlocal focus, W, H, drag_splitter, main_overlay_camera
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
            hit = dashboard_hit_test((W, H), (x, y), layout_controls)
            if hit in ("cam0", "cam1"):
                main_overlay_camera = int(hit[-1])
            else:
                focus = hit

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
                                  layout_controls=layout_controls,
                                  overlay_camera=main_overlay_camera)
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
        try:
            mgr = RecordingDataManager(cfg.abspath(cfg.get("recording", "output_dir",
                                                          default="recordings")))
            pending = mgr.pending_trials()
            if pending:
                print(f"[trials] processing {len(pending)} pending trial(s) before close")
                mgr.process_pending(cfg)
        except Exception as exc:
            print("[trials] pending processing failed:", exc)
        cv2.destroyAllWindows()
