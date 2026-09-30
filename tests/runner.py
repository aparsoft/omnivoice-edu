"""Minimal runner for the unit-test modules: ``python -m tests.<module>``."""

import sys


def run_tests(namespace: dict) -> None:
    """Run every ``test_*`` function in ``namespace`` and exit non-zero on failure."""
    tests = [n for n, f in namespace.items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name in tests:
        try:
            namespace[name]()
            print(f"  ✓ {name}")
        except Exception as e:  # noqa: BLE001 - report every failure
            kind = "" if isinstance(e, AssertionError) else f"{type(e).__name__}: "
            print(f"  ✗ {name}: {kind}{e}")
            failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
