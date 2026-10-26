#!/usr/bin/env python3
"""CI helper: grade this repo's notebook against the public tests/ directory,
and exit non-zero if any public point is missing.

We do NOT use `otter check`'s own exit code: its CLI always exits 0, even on
total failure, so it cannot gate CI. Instead we grade the notebook with otter's
own execution engine and fail the job if the public score is not full.

Hidden tests are not available here, so a green run confirms the notebook runs
on CPU and the public checks pass; it does not guarantee full credit (the hidden
Gradescope tests use different inputs and tolerances).
"""
import glob
import sys

from otter.execute import grade_notebook

NOTEBOOK = "master.ipynb"
TEST_DIR = "tests"


def main() -> int:
    tests_glob = sorted(glob.glob(f"{TEST_DIR}/*.py"))
    if not tests_glob:
        print(f"No test files found under {TEST_DIR}/", file=sys.stderr)
        return 1

    results = grade_notebook(
        NOTEBOOK,
        tests_glob=tests_glob,
        test_dir=TEST_DIR,
        ignore_errors=True,
    )

    print(results)
    print(f"\nScore: {results.total}/{results.possible} points on public tests")

    if results.possible == 0 or results.total < results.possible:
        print("\nFAILED: one or more public test cases did not pass.", file=sys.stderr)
        return 1

    print("\nAll public test cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
