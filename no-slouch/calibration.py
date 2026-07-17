"""Baseline capture: record the user's "good posture" angle and persist it."""

import statistics
from typing import Callable, Optional

from config import Config
from pose import PoseGeometry

NUM_CALIBRATION_FRAMES = 30


class Calibrator:
    """Collects angle samples one frame at a time until it has enough.

    Designed to be fed frames from an already-running capture loop (e.g. the
    UI's live feed) rather than owning its own webcam, so calibration can
    happen inline with the main detection loop.
    """

    def __init__(self, num_frames: int = NUM_CALIBRATION_FRAMES):
        self.num_frames = num_frames
        self._samples: list[float] = []
        self.active = False

    def start(self) -> None:
        self._samples = []
        self.active = True

    def cancel(self) -> None:
        self.active = False
        self._samples = []

    @property
    def progress(self) -> int:
        return len(self._samples)

    @property
    def done(self) -> bool:
        return len(self._samples) >= self.num_frames

    def feed(self, geometry: PoseGeometry) -> bool:
        """Feed one frame's geometry in. Returns True once calibration is done.

        Low-confidence frames are skipped entirely (not counted), same rule
        as the main detection loop.
        """
        if not self.active or self.done:
            return self.done

        if not geometry.confident:
            return False

        angle = geometry.angle()
        if angle is None:
            return False

        self._samples.append(angle)
        if self.done:
            self.active = False
        return self.done

    @property
    def baseline_angle(self) -> Optional[float]:
        if not self._samples:
            return None
        return statistics.median(self._samples)

    def finish_and_save(self, config: Config) -> float:
        """Persist the computed baseline angle to config.json."""
        baseline = self.baseline_angle
        if baseline is None:
            raise RuntimeError("No samples collected; cannot calibrate.")
        config.update(baseline_angle=round(baseline, 2))
        return baseline


def run_calibration_cli(camera_index: int = 0) -> float:
    """Standalone CLI calibration flow (build step 2 smoke test).

    Opens its own webcam, waits for the user to press 'c' with good posture,
    captures NUM_CALIBRATION_FRAMES samples, saves to config.json, and
    returns the baseline angle.
    """
    import cv2

    from pose import PoseEstimator, draw_skeleton, NEUTRAL_COLOR

    config = Config()
    estimator = PoseEstimator()
    cap = cv2.VideoCapture(camera_index)
    calibrator = Calibrator()

    print("Sit up straight and look at the camera, then press 'c' to calibrate.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            geometry = estimator.process(frame)
            draw_skeleton(frame, geometry, NEUTRAL_COLOR)

            if calibrator.active:
                done = calibrator.feed(geometry)
                cv2.putText(
                    frame,
                    f"Calibrating... {calibrator.progress}/{calibrator.num_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                if done:
                    baseline = calibrator.finish_and_save(config)
                    print(f"Calibration complete. baseline_angle={baseline:.2f}")
                    break
            else:
                cv2.putText(
                    frame,
                    "Press 'c' to calibrate",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and not calibrator.active:
                calibrator.start()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    return calibrator.baseline_angle


if __name__ == "__main__":
    run_calibration_cli()
