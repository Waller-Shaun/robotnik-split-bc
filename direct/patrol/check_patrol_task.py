"""Focused checks for hold timing, observation/truth separation and stage resets."""
import unittest
from patrol_task_state import PatrolMonitor


class PatrolTaskChecks(unittest.TestCase):
    def test_hold_must_be_continuous(self):
        m=PatrolMonitor(1.)
        self.assertFalse(m.update(0.,.005,.005))
        self.assertFalse(m.update(.6,.02,.02))
        self.assertFalse(m.update(.7,.005,.005))
        self.assertFalse(m.update(1.2,.005,.005))
        self.assertTrue(m.update(1.7,.005,.005))

    def test_truth_does_not_advance_task(self):
        m=PatrolMonitor(1.)
        self.assertFalse(m.update(0.,.02,.005))
        self.assertFalse(m.update(1.1,.02,.005))
        self.assertTrue(m.report()['true_hold_passed'])

    def test_observed_success_can_fail_truth_scoring(self):
        m=PatrolMonitor(1.)
        self.assertFalse(m.update(0.,.005,.02))
        self.assertTrue(m.update(1.1,.005,.02))
        self.assertFalse(m.report()['true_hold_passed'])

    def test_new_target_starts_new_hold_clock(self):
        a=PatrolMonitor(1.);a.update(0.,.005,.005)
        self.assertTrue(a.update(1.,.005,.005))
        b=PatrolMonitor(1.)
        self.assertFalse(b.update(0.,.005,.005))
        self.assertEqual(b.report()['true_longest_continuous_hold_s'],0.)

    def test_threshold_strict_and_first_reach_preserved(self):
        m=PatrolMonitor(0.)
        self.assertFalse(m.update(0.,.01,.01))
        self.assertTrue(m.update(.1,.009,.009))
        m.update(.2,.02,.02)
        self.assertEqual(m.report()['true_first_reach_elapsed_s'],.1)

    def test_application_5cm_threshold_used_for_both_observation_and_truth(self):
        m=PatrolMonitor(1.,.05)
        self.assertFalse(m.update(0.,.04,.045))
        self.assertTrue(m.update(1.,.04,.045))
        self.assertTrue(m.report()['true_hold_passed'])
        n=PatrolMonitor(1.,.05)
        n.update(0.,.04,.051)
        self.assertTrue(n.update(1.,.04,.051))
        self.assertFalse(n.report()['true_hold_passed'])


if __name__=='__main__':unittest.main()
