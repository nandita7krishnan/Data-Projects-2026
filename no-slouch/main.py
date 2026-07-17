"""Entry point: launches the tkinter UI and runs the detection loop.

The webcam + pose + detection logic runs continuously in a background
thread so the UI stays responsive and detection keeps working while the
window is minimized to tray. The background thread never touches tkinter
directly -- it writes the latest frame/status into a lock-protected dict
that the UI polls on a timer (`root.after`).
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from alerter import Alerter, SessionLog
from calibration import Calibrator
from config import Config
from detector import PostureState, SlouchDetector
from pose import GOOD_POSTURE_COLOR, NEUTRAL_COLOR, SLOUCH_COLOR, PoseEstimator, draw_skeleton

VIDEO_WIDTH = 480
VIDEO_HEIGHT = 360
UI_REFRESH_MS = 33  # ~30fps


class PostureApp:
    def __init__(self):
        self.config = Config()
        self.pose_estimator = PoseEstimator()
        self.session_log = SessionLog()
        self.alerter = Alerter(
            cooldown_seconds=self.config["alert_cooldown_seconds"],
            session_log=self.session_log,
        )
        self.calibrator = Calibrator()

        self.detector: Optional[SlouchDetector] = None
        if self.config.is_calibrated:
            self._build_detector()

        self.cap: Optional[cv2.VideoCapture] = None
        self._camera_index = self.config["camera_index"]
        self._pending_camera_index: Optional[int] = None
        self._open_camera(self._camera_index)

        self._lock = threading.Lock()
        self._shared = {"photo_frame": None, "status": {}}
        self._running = True
        self._last_tick = time.monotonic()

        self._tray_icon = None
        self._tray_thread = None

        self._build_ui()

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

        self.root.after(UI_REFRESH_MS, self._refresh_ui)

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------
    def _open_camera(self, index: int) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)
        self._camera_index = index

    def request_camera_index(self, index: int) -> None:
        self._pending_camera_index = index

    # ------------------------------------------------------------------
    # Detector lifecycle
    # ------------------------------------------------------------------
    def _build_detector(self) -> None:
        self.detector = SlouchDetector(
            baseline_angle=self.config["baseline_angle"],
            slouch_threshold_degrees=self.config["slouch_threshold_degrees"],
            grace_period_seconds=self.config["grace_period_seconds"],
        )

    def start_calibration(self) -> None:
        self.calibrator.start()

    # ------------------------------------------------------------------
    # Background detection loop (runs in its own thread, never touches Tk)
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            if self._pending_camera_index is not None:
                self._open_camera(self._pending_camera_index)
                self._pending_camera_index = None

            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                self._publish(None, {"error": "Camera unavailable", "state": "no_camera"})
                continue

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            geometry = self.pose_estimator.process(frame)
            angle = geometry.angle() if geometry.confident else None

            now = time.monotonic()
            dt = max(0.0, now - self._last_tick)
            self._last_tick = now

            status = {"error": None}

            if self.calibrator.active:
                done = self.calibrator.feed(geometry)
                status["state"] = "calibrating"
                status["calibration_progress"] = self.calibrator.progress
                status["calibration_target"] = self.calibrator.num_frames
                color = NEUTRAL_COLOR
                if done:
                    baseline = self.calibrator.finish_and_save(self.config)
                    if self.detector is None:
                        self._build_detector()
                    else:
                        self.detector.set_baseline(baseline)
                    status["just_calibrated"] = True
            elif self.detector is None:
                status["state"] = "uncalibrated"
                color = NEUTRAL_COLOR
            else:
                result = self.detector.update(angle, dt)
                self.session_log.record_tick(result.state, dt)

                if result.should_alert:
                    self.alerter.fire()

                status["state"] = result.state.value
                status["current_angle"] = result.current_angle
                status["baseline_angle"] = self.detector.baseline_angle
                status["slouch_timer"] = result.slouch_timer
                status["grace_period"] = result.grace_period_seconds
                status["cooldown_remaining"] = self.alerter.seconds_until_ready()

                color = {
                    PostureState.GOOD: GOOD_POSTURE_COLOR,
                    PostureState.SLOUCHING: SLOUCH_COLOR,
                    PostureState.NO_PERSON: NEUTRAL_COLOR,
                }[result.state]

            draw_skeleton(frame, geometry, color)
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            frame = cv2.flip(frame, 1)  # mirror for a natural "looking at self" view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._publish(rgb, status)

    def _publish(self, rgb_frame, status: dict) -> None:
        with self._lock:
            self._shared["photo_frame"] = rgb_frame
            self._shared["status"] = status

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Posture Correction")
        self.root.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)
        self._default_bg = self.root.cget("bg")

        self.banner_var = tk.StringVar(value="")
        self.banner_label = tk.Label(
            self.root, textvariable=self.banner_var, fg="#b8860b", font=("Helvetica", 11, "bold")
        )
        self.banner_label.pack(pady=(8, 0))

        self.shrimp_var = tk.StringVar(value="")
        self.shrimp_label = tk.Label(
            self.root, textvariable=self.shrimp_var, font=("Helvetica", 22, "bold")
        )
        self.shrimp_label.pack(pady=(4, 0))

        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)

        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=(0, 8))

        self.angle_var = tk.StringVar(value="Angle: -- / Baseline: --")
        tk.Label(info_frame, textvariable=self.angle_var, font=("Helvetica", 11)).pack()

        self.timer_var = tk.StringVar(value="")
        tk.Label(info_frame, textvariable=self.timer_var, font=("Helvetica", 11)).pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        tk.Button(button_frame, text="Calibrate", command=self.start_calibration).grid(row=0, column=0, padx=4)
        tk.Button(button_frame, text="Settings", command=self.open_settings).grid(row=0, column=1, padx=4)
        tk.Button(button_frame, text="Minimize to Tray", command=self.minimize_to_tray).grid(row=0, column=2, padx=4)
        tk.Button(button_frame, text="Quit", command=self.quit_app).grid(row=0, column=3, padx=4)

    def _refresh_ui(self) -> None:
        if not self._running:
            return

        with self._lock:
            rgb_frame = self._shared["photo_frame"]
            status = dict(self._shared["status"])

        if rgb_frame is not None:
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # keep a reference, avoid GC

        self._update_status_text(status)
        self.root.after(UI_REFRESH_MS, self._refresh_ui)

    def _update_status_text(self, status: dict) -> None:
        state = status.get("state")

        if status.get("error"):
            self.banner_var.set(status["error"])
            self.angle_var.set("")
            self.timer_var.set("")
            return

        if state == "calibrating":
            progress = status.get("calibration_progress", 0)
            target = status.get("calibration_target", 30)
            self.banner_var.set(f"Calibrating... hold still, sitting up straight ({progress}/{target})")
            self.angle_var.set("")
            self.timer_var.set("")
            return

        if state == "uncalibrated":
            self.banner_var.set('Sit up straight and look at the camera, then press "Calibrate".')
            self.angle_var.set("")
            self.timer_var.set("")
            return

        self.banner_var.set("")

        if status.get("just_calibrated"):
            self.banner_var.set("Calibration complete!")

        if state == "no_person":
            self.angle_var.set("No person detected -- detection paused")
            self.timer_var.set("")
            self._set_shrimp_alert(False)
            return

        current_angle = status.get("current_angle")
        baseline_angle = status.get("baseline_angle")
        if current_angle is not None and baseline_angle is not None:
            self.angle_var.set(f"Angle: {current_angle:.1f}\u00b0 / Baseline: {baseline_angle:.1f}\u00b0")

        slouch_timer = status.get("slouch_timer", 0.0)
        grace_period = status.get("grace_period", 0.0)
        if state == "slouching":
            self.timer_var.set(f"Slouching for {slouch_timer:.0f}s / {grace_period:.0f}s")
            self._set_shrimp_alert(True)
        else:
            cooldown = status.get("cooldown_remaining", 0.0)
            if cooldown > 0:
                self.timer_var.set(f"Good posture (next alert available in {cooldown:.0f}s)")
            else:
                self.timer_var.set("Good posture")
            self._set_shrimp_alert(False)

    def _set_shrimp_alert(self, active: bool) -> None:
        if not active:
            if self.shrimp_var.get():
                self.shrimp_var.set("")
                self.root.configure(bg=self._default_bg)
            return

        # Flash between two colors ~twice a second for a proper alarm feel.
        flash_on = int(time.time() * 2) % 2 == 0
        color = "#ff3b30" if flash_on else "#ff9500"
        self.shrimp_var.set("\U0001F990 SHRIMP ALERT! \U0001F990")
        self.shrimp_label.configure(fg=color)
        self.root.configure(bg=color)

    # ------------------------------------------------------------------
    # Settings panel
    # ------------------------------------------------------------------
    def open_settings(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.resizable(False, False)

        threshold_var = tk.DoubleVar(value=self.config["slouch_threshold_degrees"])
        grace_var = tk.DoubleVar(value=self.config["grace_period_seconds"])
        cooldown_var = tk.DoubleVar(value=self.config["alert_cooldown_seconds"] / 60.0)
        camera_var = tk.StringVar(value=str(self.config["camera_index"]))

        def labeled_slider(row, text, var, frm, to, resolution, unit):
            tk.Label(win, text=text).grid(row=row, column=0, sticky="w", padx=10, pady=6)
            scale = tk.Scale(
                win, variable=var, from_=frm, to=to, resolution=resolution,
                orient="horizontal", length=220,
            )
            scale.grid(row=row, column=1, padx=10)
            tk.Label(win, text=unit).grid(row=row, column=2, sticky="w")

        labeled_slider(0, "Slouch threshold (degrees)", threshold_var, 5, 40, 1, "deg")
        labeled_slider(1, "Grace period (seconds)", grace_var, 3, 300, 1, "sec")
        labeled_slider(2, "Alert cooldown (minutes)", cooldown_var, 0, 30, 0.5, "min")

        tk.Label(win, text="Camera index").grid(row=3, column=0, sticky="w", padx=10, pady=6)
        camera_menu = ttk.Combobox(win, textvariable=camera_var, values=["0", "1", "2"], width=5, state="readonly")
        camera_menu.grid(row=3, column=1, sticky="w", padx=10)

        def save_and_close():
            new_threshold = threshold_var.get()
            new_grace = grace_var.get()
            new_cooldown_seconds = cooldown_var.get() * 60.0
            new_camera_index = int(camera_var.get())

            self.config.update(
                slouch_threshold_degrees=new_threshold,
                grace_period_seconds=new_grace,
                alert_cooldown_seconds=new_cooldown_seconds,
                camera_index=new_camera_index,
            )

            if self.detector is not None:
                self.detector.slouch_threshold_degrees = new_threshold
                self.detector.grace_period_seconds = new_grace
                self.detector.reset()

            self.alerter.cooldown_seconds = new_cooldown_seconds

            if new_camera_index != self._camera_index:
                self.request_camera_index(new_camera_index)

            win.destroy()

        button_row = tk.Frame(win)
        button_row.grid(row=4, column=0, columnspan=3, pady=10)
        tk.Button(button_row, text="Save", command=save_and_close).pack(side="left", padx=6)
        tk.Button(button_row, text="Cancel", command=win.destroy).pack(side="left", padx=6)

    # ------------------------------------------------------------------
    # Tray icon
    # ------------------------------------------------------------------
    def minimize_to_tray(self) -> None:
        self.root.withdraw()
        if self._tray_thread is None or not self._tray_thread.is_alive():
            self._tray_thread = threading.Thread(target=self._run_tray, daemon=True)
            self._tray_thread.start()

    def _run_tray(self) -> None:
        import pystray
        from PIL import Image as PILImage, ImageDraw

        size = 64
        icon_image = PILImage.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(icon_image)
        draw.ellipse((8, 8, size - 8, size - 8), fill=(30, 130, 76))

        def on_show(icon, item):
            self.root.after(0, self._restore_from_tray)

        def on_recalibrate(icon, item):
            self.root.after(0, self._restore_from_tray)
            self.root.after(100, self.start_calibration)

        def on_quit(icon, item):
            icon.stop()
            self.root.after(0, self.quit_app)

        menu = pystray.Menu(
            pystray.MenuItem("Show", on_show, default=True),
            pystray.MenuItem("Recalibrate", on_recalibrate),
            pystray.MenuItem("Quit", on_quit),
        )
        self._tray_icon = pystray.Icon("posture_app", icon_image, "Posture Correction", menu)
        self._tray_icon.run()

    def _restore_from_tray(self) -> None:
        self.root.deiconify()
        self.root.lift()
        if self._tray_icon is not None:
            self._tray_icon.stop()
            self._tray_icon = None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def quit_app(self) -> None:
        self._running = False

        if self._tray_icon is not None:
            try:
                self._tray_icon.stop()
            except Exception:
                pass

        if self.cap is not None:
            self.cap.release()
        self.pose_estimator.close()

        summary = self.session_log.finalize_and_save()
        try:
            messagebox.showinfo(
                "Session Summary",
                (
                    f"Duration: {summary['duration_seconds'] / 60.0:.1f} min\n"
                    f"Slouch alerts: {summary['alert_count']}\n"
                    f"Good posture: {summary['good_posture_percent']:.0f}%"
                ),
            )
        except Exception:
            print(summary)

        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = PostureApp()
    app.run()
