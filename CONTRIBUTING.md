# Contributing

Thank you for helping improve Trend Emperor BTC Sentinel.

## Scope

Contributions must use unauthenticated public market data and keep the project research-only. Do not add authenticated exchange clients, private endpoints, order execution, account queries, credentials, generated datasets, or operational records.

## Before opening a pull request

1. Open or reference an issue for non-trivial changes.
2. Add or update focused tests.
3. Run `python -m pytest -q`.
4. Run `python scripts/verify_public_tree.py .`.
5. Explain the behavior change, data-source assumptions, and security impact.

Maintainers triage issues for reproducibility, correctness, scope, and security. Small, reviewable changes are preferred. Never paste credentials, private account data, or sensitive logs into an issue or pull request.
