"""Freshness / staleness scoring — pure functions over ChunkRecord.freshness dicts.

Age uses source_fetched_at (falls back to ingested_at). Unknown/unparseable timestamps
yield age None → treated as NOT stale (never downgrade on missing data).
"""
from __future__ import annotations

from datetime import datetime, timezone


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_days(freshness: dict, now: datetime) -> float | None:
    ts = freshness.get("source_fetched_at") or freshness.get("ingested_at")
    dt = _parse(ts)
    if dt is None:
        return None
    return (now - dt).total_seconds() / 86400.0


def ttl_days(freshness: dict, config) -> int:
    val = freshness.get("ttl_days")
    return int(val) if val is not None else config.default_ttl_days


def is_stale(freshness: dict, config, now: datetime) -> bool:
    age = age_days(freshness, now)
    if age is None:
        return False
    return age > ttl_days(freshness, config)


def freshness_note(freshness: dict, now: datetime) -> str:
    age = age_days(freshness, now)
    if age is None:
        return ""
    return f"last verified {int(age)} days ago"


def _build_pipeline():
    from rag import RagConfig, RagPipeline
    return RagPipeline(RagConfig.from_env())


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="rag.freshness", description="Freshness sweep")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sweep = sub.add_parser("sweep", help="find + re-ingest stale sources")
    sweep.add_argument("--dry-run", action="store_true", help="report only, don't re-ingest")
    args = parser.parse_args(argv)

    if args.cmd == "sweep":
        pipeline = _build_pipeline()
        report = pipeline.sweep_stale(dry_run=args.dry_run)
        print(f"Checked {report['checked']} source(s) — {len(report['stale'])} stale.")
        for key in report["stale"]:
            mark = "would refresh" if args.dry_run else (
                "refreshed" if key in report["refreshed"] else "skipped")
            print(f"  • {key}  [{mark}]")


if __name__ == "__main__":
    main()
