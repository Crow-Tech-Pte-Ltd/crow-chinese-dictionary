"""Read a budget ledger or explicitly requeue a definitive rejected request.

Requeue never deletes attempts, refunds reservations, resets usage, changes a
fingerprint or raises a retry/budget limit. Unknown-charge, successful, invalid
schema and provider-anomaly outcomes are not eligible for operator requeue.
No credentials or provider calls are involved.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import time

from .generate import Ledger, canonical, money

REJECTED_CODES = frozenset((401, 402, 403, 404, 422))
FAILED_STATUSES = frozenset(("failed", "review_failed"))
UNSAFE_HISTORY = frozenset(("reserved", "requesting", "quarantined", "machine_draft", "abstained",
    "schema_invalid", "review_completed", "review_invalid", "source_imported"))


class LedgerOperationError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


@contextmanager
def connection(path, *, readonly):
    path = Path(path).resolve()
    if not path.is_file():
        raise LedgerOperationError("ledger_not_found")
    db = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True, timeout=30)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        required = {"budget": {"id", "max_micro", "max_calls", "max_tokens", "spent_micro", "calls", "tokens_reserved", "halted"},
            "requests": {"fingerprint", "job_id", "status", "updated", "issues_json"},
            "attempts": {"id", "fingerprint", "attempt_no", "status", "error_code", "reserved_micro", "settled_micro"}}
        for table, columns in required.items():
            if not columns <= {row[1] for row in db.execute("PRAGMA table_info(" + table + ")")}:
                raise LedgerOperationError("unsupported_ledger_schema")
        yield db
    finally:
        db.close()


def summary(path):
    with connection(path, readonly=True) as db:
        row = db.execute("SELECT * FROM budget WHERE id=1").fetchone()
        if row is None:
            raise LedgerOperationError("missing_budget")
        statuses = dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status"))
        attempts = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        provider_budgets = Ledger.provider_summary(db) if "provider_budgets" in tables else {}
        rejected = []
        for request in db.execute("SELECT fingerprint,job_id,status FROM requests WHERE status IN ('failed','review_failed') ORDER BY job_id,fingerprint"):
            last = db.execute("SELECT id,attempt_no,status,error_code FROM attempts WHERE fingerprint=? ORDER BY attempt_no DESC,id DESC LIMIT 1", (request["fingerprint"],)).fetchone()
            try:
                code = int(last["error_code"]) if last else 0
            except (ValueError, TypeError):
                code = 0
            if last and last["status"] in FAILED_STATUSES and code in REJECTED_CODES:
                rejected.append({"jobId": request["job_id"], "fingerprint": request["fingerprint"],
                    "httpCode": code, "attemptCount": last["attempt_no"]})
        return {"maxUsd": money(row["max_micro"]), "reservedOrKnownUsd": money(row["spent_micro"]),
            "calls": row["calls"], "maxCalls": row["max_calls"], "tokensConservativelyReserved": row["tokens_reserved"],
            "maxTotalTokens": row["max_tokens"], "halted": bool(row["halted"]), "statuses": statuses,
            "attempts": attempts, "definitivelyRejectedRequests": rejected, "providerBudgets": provider_budgets,
            "networkRequests": 0, "credentialsRead": False}


def requeue_rejected(path, job_id, fingerprint=None):
    if not isinstance(job_id, str) or not job_id or len(job_id) > 200:
        raise LedgerOperationError("valid_job_id_required")
    with connection(path, readonly=False) as db:
        db.execute("BEGIN IMMEDIATE")
        budget = db.execute("SELECT halted FROM budget WHERE id=1").fetchone()
        if budget is None:
            raise LedgerOperationError("missing_budget")
        if budget["halted"]:
            raise LedgerOperationError("halted_ledger_requires_separate_investigation")
        query = "SELECT fingerprint,job_id,status,issues_json FROM requests WHERE job_id=?"
        args = [job_id]
        if fingerprint is not None:
            query += " AND fingerprint=?"
            args.append(fingerprint)
        rows = db.execute(query, args).fetchall()
        if not rows:
            raise LedgerOperationError("request_not_found")
        if len(rows) != 1:
            raise LedgerOperationError("ambiguous_job_requires_fingerprint")
        request = rows[0]
        if request["status"] not in FAILED_STATUSES:
            raise LedgerOperationError("request_is_not_a_definitive_rejection")
        history = db.execute("SELECT id,attempt_no,status,error_code FROM attempts WHERE fingerprint=? ORDER BY attempt_no,id", (request["fingerprint"],)).fetchall()
        if not history:
            raise LedgerOperationError("missing_request_attempt")
        if any(row["status"] in UNSAFE_HISTORY for row in history):
            raise LedgerOperationError("unsafe_or_unknown_charge_history")
        last = history[-1]
        try:
            code = int(last["error_code"])
        except (ValueError, TypeError):
            code = 0
        if last["status"] not in FAILED_STATUSES or code not in REJECTED_CODES:
            raise LedgerOperationError("latest_attempt_is_not_an_allowed_rejection")
        try:
            issues = json.loads(request["issues_json"] or "[]")
        except (ValueError, TypeError):
            raise LedgerOperationError("malformed_request_issues") from None
        if (not isinstance(issues, list) or any(not isinstance(item, str) for item in issues)
                or any(any(word in item for word in ("uncertain", "unknown", "anomaly", "exceeded", "mismatch", "credential_echo")) for item in issues)):
            raise LedgerOperationError("unsafe_or_unknown_charge_history")
        db.execute("""CREATE TABLE IF NOT EXISTS ledger_operator_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,created REAL NOT NULL,event TEXT NOT NULL,
            job_id TEXT NOT NULL,fingerprint TEXT NOT NULL,previous_status TEXT NOT NULL,
            latest_attempt_id INTEGER NOT NULL,http_code INTEGER NOT NULL)""")
        db.execute("INSERT INTO ledger_operator_events(created,event,job_id,fingerprint,previous_status,latest_attempt_id,http_code) VALUES(?,'requeue_definitive_rejection',?,?,?,?,?)",
            (time.time(), job_id, request["fingerprint"], request["status"], last["id"], code))
        if "operator_requeued_definitive_rejection" not in issues:
            issues.append("operator_requeued_definitive_rejection")
        db.execute("UPDATE requests SET status='retryable',updated=?,issues_json=? WHERE fingerprint=?",
            (time.time(), canonical(issues), request["fingerprint"]))
        db.commit()
        return {"requeued": True, "jobId": job_id, "fingerprint": request["fingerprint"],
            "previousStatus": request["status"], "httpCode": code, "attemptCount": len(history),
            "usageAndReservationsRetained": True, "runnerRetryLimitStillApplies": True,
            "networkRequests": 0, "credentialsRead": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--summary", action="store_true")
    actions.add_argument("--requeue-rejected", action="store_true",
        help="Explicit operator action for one job after fixing credentials/configuration")
    parser.add_argument("--job-id", help="Exactly one job; required for requeue")
    parser.add_argument("--fingerprint", help="Required when the job has multiple request fingerprints")
    args = parser.parse_args(argv)
    try:
        if args.summary:
            if args.job_id or args.fingerprint:
                raise LedgerOperationError("summary_does_not_take_request_identifiers")
            value = summary(args.ledger)
        else:
            if not args.job_id:
                raise LedgerOperationError("job_id_required_for_requeue")
            value = requeue_rejected(args.ledger, args.job_id, args.fingerprint)
        print(canonical(value))
        return 0
    except LedgerOperationError as error:
        print(canonical({"error": error.code, "networkRequests": 0, "credentialsRead": False}), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, sqlite3.Error):
        print(canonical({"error": "ledger_operation_failed", "networkRequests": 0, "credentialsRead": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
