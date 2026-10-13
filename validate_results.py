"""Post-run validation of test_results.csv against munder_difflin.db.

Run after project_starter.py:  python validate_results.py

Checks:
  1. No customer reply contains a blocked phrase (stock counts, internal
     details, or offers of services the company did not perform).
  2. The totals the replies state for each date equal the sales recorded in
     the database for that date.
  3. Flags requests where nothing was sold but the cash balance still moved
     (stock bought for an order that then did not sell).
  4. Rubric minimums: at least three requests change the cash balance, at
     least three are fulfilled (something sold), and not all are fulfilled.

Exits with status 1 if check 1, 2 or 4 fails; check 3 is reported as a flag.
"""
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Pattern, Tuple

TOTAL_CHARGED = re.compile(r"Total charged: \$([\d,]+\.\d{2})")


def _stated_total(reply: str) -> float:
    """The 'Total charged' a reply states, or 0.0 if it states none (nothing sold)."""
    match = TOTAL_CHARGED.search(reply)
    return float(match.group(1).replace(",", "")) if match else 0.0


def validate(results_path: str, db_path: str,
             blocked_patterns: List[Pattern]) -> Tuple[List[str], List[str]]:
    """Return (problems, flags) for one evaluation run."""
    with open(results_path, encoding="utf-8-sig", newline="") as results_file:
        rows = list(csv.DictReader(results_file))
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    problems, flags = [], []

    # 1. Blocked phrases in any reply.
    for row in rows:
        hits = sorted({match.group(0) for pattern in blocked_patterns
                       for match in pattern.finditer(row["response"])})
        if hits:
            problems.append(f"request {row['request_id']}: reply contains blocked phrases {hits}")

    # 2. Stated totals vs recorded sales, per date (several requests can share a date).
    stated_by_date: Dict[str, float] = defaultdict(float)
    for row in rows:
        stated_by_date[row["request_date"]] += _stated_total(row["response"])
    for date, stated in sorted(stated_by_date.items()):
        recorded = connection.execute(
            "SELECT COALESCE(SUM(price), 0) FROM transactions "
            "WHERE transaction_type = 'sales' AND item_name IS NOT NULL AND transaction_date = ?",
            (date,),
        ).fetchone()[0]
        if abs(recorded - stated) >= 0.005:
            problems.append(f"{date}: replies state ${stated:,.2f} charged but ${recorded:,.2f} of sales were recorded")

    # 3. Cash movement per request. The harness reports cash after each request,
    # so the change from the previous row is that request's sales minus stock bought.
    first_date = min(row["request_date"] for row in rows)
    previous_cash = connection.execute(
        "SELECT COALESCE(SUM(CASE WHEN transaction_type = 'sales' THEN price ELSE -price END), 0) "
        "FROM transactions WHERE transaction_date < ?",
        (first_date,),
    ).fetchone()[0]
    cash_changes = fulfilled = 0
    for row in rows:
        cash = float(row["cash_balance"])
        change, previous_cash = round(cash - previous_cash, 2), cash
        stated = _stated_total(row["response"])
        cash_changes += abs(change) >= 0.005
        fulfilled += stated > 0
        if stated == 0 and abs(change) >= 0.005:
            flags.append(f"request {row['request_id']}: nothing sold but cash moved by ${change:,.2f} (restocking)")

    # 4. Rubric minimums for the evaluation run.
    print(f"Requests: {len(rows)} | cash changed: {cash_changes} | fulfilled: {fulfilled} | "
          f"not fulfilled: {len(rows) - fulfilled}")
    if cash_changes < 3:
        problems.append(f"only {cash_changes} requests changed the cash balance (rubric needs 3+)")
    if fulfilled < 3:
        problems.append(f"only {fulfilled} requests were fulfilled (rubric needs 3+)")
    if fulfilled == len(rows):
        problems.append("every request was fulfilled (rubric needs some unfulfilled with reasons)")
    connection.close()
    return problems, flags


def main() -> int:
    """Validate the files written by the last run of project_starter.py."""
    # Same patterns the system uses before a reply is sent.
    from project_starter import CUSTOMER_BLOCKLIST, _CUSTOMER_OFFER_PATTERNS, _INTERNAL_DETAIL_PATTERNS

    problems, flags = validate("test_results.csv", "munder_difflin.db",
                               CUSTOMER_BLOCKLIST + _CUSTOMER_OFFER_PATTERNS + _INTERNAL_DETAIL_PATTERNS)
    for flag in flags:
        print(f"FLAG    {flag}")
    for problem in problems:
        print(f"PROBLEM {problem}")
    print("All checks passed." if not problems else f"{len(problems)} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
