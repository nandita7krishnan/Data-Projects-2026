# no-slouch

A lightweight, fully local posture-correction desktop app. Uses MediaPipe
Pose over your webcam to track the ear-shoulder-hip angle, compares it
against a calibrated baseline, and nags you (system notification + chime)
if you've been slouching continuously past a grace period. Everything runs
on-device -- no data leaves your machine, no cloud, no LLM.

## Setup

**Don't use the macOS system Python** (`/usr/bin/python3`, the Command Line
Tools one) for the venv. It links Apple's ancient bundled Tcl/Tk 8.5, which
hard-crashes (`Tcl_Panic` / `abort()` in `TkpInit`) on current macOS (26+) --
you'll see `macOS 26 (2600) or later required, have instead 16 (1600)!` in
the crash log. Use a Homebrew Python with a modern Tk instead:

```bash
brew install python-tk@3.12   # pulls in python@3.12 + a working tcl-tk

# if you hit "Symbol not found: _XML_SetAllocTrackerActivationThreshold"
# importing pyexpat, it means the bottle's pyexpat.so was linked against a
# newer system libexpat than your macOS point release ships. Fix:
brew install expat
SO=$(find /opt/homebrew/Cellar/python@3.12 -name "pyexpat.cpython-*.so")
install_name_tool -change /usr/lib/libexpat.1.dylib \
  "$(brew --prefix expat)/lib/libexpat.1.dylib" "$SO"
codesign --force --sign - "$SO"   # required after install_name_tool or dyld will refuse to load it

cd no-slouch
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

> **Note:** `mediapipe` is pinned to `0.10.14`. Versions `0.10.18+` dropped
> the legacy `mp.solutions.pose` API this app is built on, so don't bump
> that pin without also rewriting `pose.py` against the newer Tasks API.

On macOS, the first run will prompt for Camera and Notification permissions
-- allow both. If the camera prompt doesn't appear or capture fails, check
System Settings > Privacy & Security > Camera and manually enable it for
Terminal/your terminal app.

## First run

No `config.json` means you haven't calibrated yet. Sit up straight, look at
the camera, and click **Calibrate**. It captures 30 frames and stores the
median ear-shoulder-hip angle as your baseline in `config.json`. Recalibrate
any time via the **Calibrate** button or the tray menu.

## How detection works

- Every frame, the ear/shoulder midpoints are extracted and the angle of
  the ear-shoulder line relative to vertical is computed (measured at the
  shoulder). This is deliberately *not* anchored on the hip: typical
  webcam framing (laptop/monitor cam) rarely has hips in frame, and
  mediapipe still reports a plausible-but-fabricated hip position with
  moderate confidence even when it's off-screen -- that hallucinated point
  is stable/insensitive to real shoulder movement and ends up masking
  shoulder-drop slouches that don't also involve the head tilting forward.
  Anchoring on vertical instead makes the angle directly sensitive to
  shoulder position on its own, independent of head tilt.
- If `baseline_angle - current_angle > threshold` (default 15°), you're
  slouching.
- Frames with low landmark confidence (<0.6 for ears/shoulders) are
  discarded, not counted as slouching.
- Continuous slouching for `grace_period_seconds` (default 5s) fires an
  alert, gated by a cooldown (default 5 min) so it doesn't nag.
- Stepping away from the desk pauses detection entirely.

## Files

| File | Purpose |
|---|---|
| `pose.py` | MediaPipe wrapper: landmark extraction, angle math, skeleton overlay |
| `calibration.py` | Baseline capture flow |
| `detector.py` | Slouch/grace-period state machine (+ unit tests) |
| `alerter.py` | Notification/chime firing, cooldown, session logging |
| `config.py` | `config.json` load/save |
| `main.py` | tkinter UI, tray icon, background detection loop |

Run the detector's unit tests with:

```bash
python -m unittest detector -v
```

`session_log.json` accumulates one summary entry (duration, alert count,
% good posture) per session, appended on quit.

## Minimize to tray

Closing the window or clicking **Minimize to Tray** hides the window but
keeps detection running in the background. Right-click the tray icon for
Show / Recalibrate / Quit.
