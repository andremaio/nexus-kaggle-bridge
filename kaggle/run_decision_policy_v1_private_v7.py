#!/usr/bin/env python3
from __future__ import annotations

import eval_decision_policy_v1_private_v7 as evaluator


if __name__ == '__main__':
    evaluator.private_v7()
    # The evaluator report is the source of truth. Keep the private kernel itself
    # successful so aggregate evidence can always be collected; the outer workflow
    # fails closed when report.gates.passed is false.
