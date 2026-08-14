#!/usr/bin/env python3
from __future__ import annotations

import check_qwen4b_training_isolation as gate


CORRECTED_V5_COMMIT = "f6b583b9c0391a9c939b66fa52b777ec8d7f2f3c"
CORRECTED_V5_BLOB = "f8c144f2c814c71837999dc4ec80b84c057c6ac4"


def main() -> int:
    gate.V5_COMMIT = CORRECTED_V5_COMMIT
    gate.EVAL_FILES["holdout_v5.jsonl"] = (CORRECTED_V5_COMMIT, CORRECTED_V5_BLOB)
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
