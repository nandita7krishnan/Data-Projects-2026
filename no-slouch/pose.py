"""MediaPipe Pose wrapper: landmark extraction and angle geometry.

Mediapipe's classic `Pose` solution already returns a single pose per frame
(the most prominent person), so the "multiple people in frame" edge case is
handled for free -- we don't need to pick a pose ourselves.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

LEFT_EAR = mp_pose.PoseLandmark.LEFT_EAR
RIGHT_EAR = mp_pose.PoseLandmark.RIGHT_EAR
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP

CONFIDENCE_THRESHOLD = 0.6
HIP_FALLBACK_THRESHOLD = 0.5

Point = Tuple[float, float]

GOOD_POSTURE_COLOR = (0, 200, 0)  # BGR green
SLOUCH_COLOR = (0, 0, 220)  # BGR red
NEUTRAL_COLOR = (180, 180, 180)


def angle_between(a: Point, b: Point, c: Point) -> float:
    """Angle at point b, formed by points a-b-c."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


@dataclass
class PoseGeometry:
    """Extracted geometry for one frame."""

    raw_landmarks: object  # mediapipe NormalizedLandmarkList, for drawing
    ear_mid: Optional[Point]
    shoulder_mid: Optional[Point]
    hip_mid: Optional[Point]
    ear_conf: float
    shoulder_conf: float
    hip_conf: float

    @property
    def person_detected(self) -> bool:
        return self.raw_landmarks is not None

    @property
    def confident(self) -> bool:
        """Whether ear + shoulder confidence clears the usable threshold."""
        return (
            self.person_detected
            and self.ear_conf >= CONFIDENCE_THRESHOLD
            and self.shoulder_conf >= CONFIDENCE_THRESHOLD
        )

    @property
    def hip_usable(self) -> bool:
        return self.hip_conf >= HIP_FALLBACK_THRESHOLD

    def angle(self) -> Optional[float]:
        """Ear-shoulder angle relative to vertical, measured at the shoulder.

        Typical webcam framing (laptop/monitor cam) rarely has the hips in
        frame at all, and mediapipe still reports a plausible-but-fabricated
        hip position with moderate confidence even when the hip is off
        screen -- that hallucinated point is stable/insensitive to real
        shoulder movement, which ends up masking pure shoulder-drop slouches
        that don't also involve the head tilting forward. Anchoring on a
        fixed vertical reference below the shoulder instead makes this
        directly sensitive to shoulder position on its own, regardless of
        whether hips are visible or of head tilt.
        """
        if not self.confident:
            return None
        vertical_ref = (self.shoulder_mid[0], self.shoulder_mid[1] + 0.3)
        return angle_between(self.ear_mid, self.shoulder_mid, vertical_ref)


class PoseEstimator:
    """Wraps mediapipe.solutions.pose with landmark extraction helpers."""

    def __init__(self, model_complexity: int = 1):
        self._pose = mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseGeometry:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return PoseGeometry(None, None, None, None, 0.0, 0.0, 0.0)

        lm = results.pose_landmarks.landmark

        def pt(idx) -> Point:
            p = lm[idx]
            return (p.x, p.y)

        def vis(idx) -> float:
            return lm[idx].visibility

        ear_conf = min(vis(LEFT_EAR), vis(RIGHT_EAR))
        shoulder_conf = min(vis(LEFT_SHOULDER), vis(RIGHT_SHOULDER))
        hip_conf = min(vis(LEFT_HIP), vis(RIGHT_HIP))

        ear_mid = _midpoint(pt(LEFT_EAR), pt(RIGHT_EAR))
        shoulder_mid = _midpoint(pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER))
        hip_mid = _midpoint(pt(LEFT_HIP), pt(RIGHT_HIP))

        return PoseGeometry(
            raw_landmarks=results.pose_landmarks,
            ear_mid=ear_mid,
            shoulder_mid=shoulder_mid,
            hip_mid=hip_mid,
            ear_conf=ear_conf,
            shoulder_conf=shoulder_conf,
            hip_conf=hip_conf,
        )

    def close(self) -> None:
        self._pose.close()


def draw_skeleton(frame_bgr: np.ndarray, geometry: PoseGeometry, color: Tuple[int, int, int]) -> None:
    """Draw the pose skeleton overlay onto the frame in-place."""
    if not geometry.person_detected:
        return
    mp_drawing.draw_landmarks(
        frame_bgr,
        geometry.raw_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2),
    )


if __name__ == "__main__":
    # Manual smoke test: print live landmark angles to console (build step 1).
    cap = cv2.VideoCapture(0)
    estimator = PoseEstimator()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            geometry = estimator.process(frame)
            if geometry.confident:
                print(f"angle={geometry.angle():.1f} "
                      f"ear_conf={geometry.ear_conf:.2f} "
                      f"shoulder_conf={geometry.shoulder_conf:.2f} "
                      f"hip_conf={geometry.hip_conf:.2f}")
            else:
                print("no confident pose")
            cv2.imshow("pose debug", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()
