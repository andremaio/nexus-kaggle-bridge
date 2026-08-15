#!/usr/bin/env python3
from __future__ import annotations

from scripts import decision_policy_v3_data as data
from scripts import train_eval_decision_policy_v1 as base
from scripts import train_eval_decision_policy_v3 as v3


def fixed_fit_candidate():
    train = v3.as_base(data.train_rows())
    calibration = v3.as_base(data.calibration_rows())
    old = tuple(base.training_examples(base.V6)) + tuple(base.training_examples(base.V7))
    combined = train + old
    weights, biases = base.fit(combined)
    temperature = base.calibrate(weights, biases, calibration)
    known = base.known_hashes(combined)
    return combined, calibration, weights, biases, temperature, known


v3.fit_candidate = fixed_fit_candidate

if __name__ == "__main__":
    raise SystemExit(v3.main())
