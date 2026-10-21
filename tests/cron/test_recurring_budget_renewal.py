"""Explicit finite-budget renewal must not bypass other terminal guards.

Uses unittest as well as pytest: installed runtime distributions may omit pytest.
"""
import unittest
from copy import deepcopy
from cron.jobs import _has_renewed_recurring_budget, _reject_terminal_activation


class RecurringBudgetRenewal(unittest.TestCase):
    def job(self, kind: object = "interval", times: object = 192,
            completed: object = 96, state: object = "completed"):
        return {"id": "renewal-fixture", "enabled": False, "state": state,
                "schedule": {"kind": kind}, "repeat": {"times": times, "completed": completed}}

    def test_explicit_finite_recurring_renewal_can_pause_and_resume(self):
        for kind in ("interval", "cron"):
            job = self.job(kind=kind)
            for state in ("paused", "scheduled"):
                updated = deepcopy(job)
                updated.update(state=state, enabled=state == "scheduled")
                _reject_terminal_activation(job, updated, job["id"])
                self.assertEqual(updated["repeat"], {"times": 192, "completed": 96})

    def test_exhausted_or_one_shot_cannot_activate(self):
        for job in (self.job(times=96), self.job(kind="once")):
            updated = dict(job, enabled=True, state="scheduled")
            with self.assertRaises(ValueError):
                _reject_terminal_activation(job, updated, job["id"])

    def test_one_shot_never_gains_renewal_exception(self):
        for kind in ("once", "at", None):
            self.assertFalse(_has_renewed_recurring_budget(self.job(kind=kind)))

    def test_non_finite_or_malformed_counters_do_not_renew(self):
        for times, completed in ((None, 96), (192, None), (True, 0),
                                 (192, False), ("192", 96), (192, "96"),
                                 (96, 96), (95, 96), (192, -1)):
            self.assertFalse(_has_renewed_recurring_budget(self.job(times=times, completed=completed)))
        job = self.job()
        for repeat in (None, [], "192"):
            job["repeat"] = repeat
            self.assertFalse(_has_renewed_recurring_budget(job))

    def test_only_completed_state_gets_this_exception(self):
        for state in ("scheduled", "paused", "error", "cancelled", None):
            self.assertFalse(_has_renewed_recurring_budget(self.job(state=state)))


if __name__ == "__main__":
    unittest.main()
