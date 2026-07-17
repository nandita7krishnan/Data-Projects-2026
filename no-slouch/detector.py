"""Slouch detection logic: baseline comparison, grace period, timer.

Cooldown / actual alert firing lives in alerter.py -- this module only
decides *whether* the user is currently slouching and whether the
continuous slouch duration has crossed the grace period.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PostureState(Enum):
    NO_PERSON = "no_person"
    GOOD = "good"
    SLOUCHING = "slouching"


@dataclass
class DetectionResult:
    state: PostureState
    current_angle: Optional[float]
    slouch_timer: float
    grace_period_seconds: float
    should_alert: bool  # grace period just crossed on this update

    @property
    def is_slouching(self) -> bool:
        return self.state == PostureState.SLOUCHING

    @property
    def person_detected(self) -> bool:
        return self.state != PostureState.NO_PERSON


class SlouchDetector:
    """Stateful detector: feed it (angle, dt) pairs each frame.

    `angle` should be None for frames that are low-confidence or have no
    person detected -- those pause detection (reset the slouch timer)
    rather than counting as slouching, per spec.
    """

    def __init__(
        self,
        baseline_angle: float,
        slouch_threshold_degrees: float = 15.0,
        grace_period_seconds: float = 60.0,
    ):
        self.baseline_angle = baseline_angle
        self.slouch_threshold_degrees = slouch_threshold_degrees
        self.grace_period_seconds = grace_period_seconds
        self.slouch_timer: float = 0.0

    def update(self, angle: Optional[float], dt: float) -> DetectionResult:
        if angle is None:
            self.slouch_timer = 0.0
            return DetectionResult(
                state=PostureState.NO_PERSON,
                current_angle=None,
                slouch_timer=0.0,
                grace_period_seconds=self.grace_period_seconds,
                should_alert=False,
            )

        is_slouching = (self.baseline_angle - angle) > self.slouch_threshold_degrees

        if is_slouching:
            self.slouch_timer += dt
        else:
            self.slouch_timer = 0.0

        should_alert = False
        if is_slouching and self.slouch_timer >= self.grace_period_seconds:
            should_alert = True
            self.slouch_timer = 0.0

        return DetectionResult(
            state=PostureState.SLOUCHING if is_slouching else PostureState.GOOD,
            current_angle=angle,
            slouch_timer=self.slouch_timer,
            grace_period_seconds=self.grace_period_seconds,
            should_alert=should_alert,
        )

    def reset(self) -> None:
        self.slouch_timer = 0.0

    def set_baseline(self, baseline_angle: float) -> None:
        self.baseline_angle = baseline_angle
        self.reset()


# --------------------------------------------------------------------------
# Unit tests on hardcoded angles (build step 3). Run with:
#   python -m unittest detector
# --------------------------------------------------------------------------
import unittest


class SlouchDetectorTests(unittest.TestCase):
    def setUp(self):
        # baseline 140, threshold 15 -> slouching when angle < 125
        self.detector = SlouchDetector(
            baseline_angle=140.0,
            slouch_threshold_degrees=15.0,
            grace_period_seconds=10.0,
        )

    def test_good_posture_no_slouch(self):
        result = self.detector.update(angle=138.0, dt=1.0)
        self.assertEqual(result.state, PostureState.GOOD)
        self.assertFalse(result.should_alert)
        self.assertEqual(result.slouch_timer, 0.0)

    def test_slight_lean_within_threshold_not_slouching(self):
        # 140 - 130 = 10 <= 15 threshold -> not slouching
        result = self.detector.update(angle=130.0, dt=1.0)
        self.assertEqual(result.state, PostureState.GOOD)

    def test_slouch_detected_but_under_grace_period(self):
        result = self.detector.update(angle=120.0, dt=5.0)
        self.assertEqual(result.state, PostureState.SLOUCHING)
        self.assertFalse(result.should_alert)
        self.assertEqual(result.slouch_timer, 5.0)

    def test_slouch_alert_fires_at_grace_period(self):
        self.detector.update(angle=120.0, dt=5.0)
        result = self.detector.update(angle=120.0, dt=5.0)
        self.assertTrue(result.should_alert)
        self.assertEqual(result.slouch_timer, 0.0)  # reset after firing

    def test_intermittent_slouch_does_not_accumulate(self):
        self.detector.update(angle=120.0, dt=9.0)  # 9s of slouching
        result = self.detector.update(angle=138.0, dt=1.0)  # brief good posture
        self.assertEqual(result.state, PostureState.GOOD)
        self.assertEqual(result.slouch_timer, 0.0)  # timer reset, no alert

        result = self.detector.update(angle=120.0, dt=9.0)  # slouch again
        self.assertFalse(result.should_alert)  # must restart from 0, not resume

    def test_no_person_pauses_and_resets_timer(self):
        self.detector.update(angle=120.0, dt=9.0)
        result = self.detector.update(angle=None, dt=5.0)
        self.assertEqual(result.state, PostureState.NO_PERSON)
        self.assertEqual(result.slouch_timer, 0.0)
        self.assertFalse(result.should_alert)

    def test_set_baseline_resets_timer(self):
        self.detector.update(angle=120.0, dt=9.0)
        self.detector.set_baseline(150.0)
        self.assertEqual(self.detector.slouch_timer, 0.0)
        self.assertEqual(self.detector.baseline_angle, 150.0)

    def test_exact_threshold_boundary_is_not_slouching(self):
        # 140 - 125 = 15, strictly must exceed threshold to count as slouching
        result = self.detector.update(angle=125.0, dt=1.0)
        self.assertEqual(result.state, PostureState.GOOD)


if __name__ == "__main__":
    unittest.main()
