"""
CLI ingestion script.
Reads data/urls.txt (or data/web_urls.txt) and all supported files in data/ → embeds → stores in ChromaDB.
"""
from __future__ import annotations
import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".txt", ".xlsx", ".xls"}
DATA_DIR = Path("data")
URLS_FILES = (DATA_DIR / "urls.txt", DATA_DIR / "web_urls.txt")


def _normalize_url(raw: str) -> str | None:
    """Repair common URL entry errors; return clean URL or None if unrecognisable."""
    url = raw.strip()
    if not url:
        return None

    # Repair broken http/https scheme: htp://, htps://, https//, https:/ etc.
    m = re.match(r"^(h[a-z]{1,5})([:/]{1,3})(.*)", url, re.IGNORECASE)
    if m:
        scheme = "https" if "s" in m.group(1).lower() else "http"
        url = f"{scheme}://{m.group(3)}"
    elif not re.match(r"^https?://", url, re.IGNORECASE):
        # No recognisable scheme — prepend https://
        url = "https://" + url

    # Validate: netloc must exist, contain a dot, have no spaces
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if not parsed.netloc or "." not in parsed.netloc or " " in parsed.netloc:
        return None

    return url


def _log_result(r: dict, label: str) -> None:
    status = r.get("status")
    title = r.get("title", label)
    chunks = r.get("chunks", 0)
    if status == "ok":
        logger.info("  ✓ %s  [%d chunks]", title, chunks)
    elif status == "skipped":
        logger.info("  ↩ %s  [unchanged, %d chunks]", title, chunks)
    elif status == "empty":
        logger.warning("  ⚠ %s  [empty document]", title)
    elif status == "no_chunks":
        logger.warning("  ⚠ %s  [no chunks produced]", title)
    else:
        logger.warning("  ? %s  [status=%s]", title, status)


def load_urls() -> list[str]:
    seen: dict[str, str] = {}  # dedup_key → first stored URL
    urls: list[str] = []

    for url_file in URLS_FILES:
        if not url_file.exists():
            continue
        for lineno, raw in enumerate(url_file.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            normalized = _normalize_url(line)
            if normalized is None:
                logger.warning("Skipping unrecognisable URL (%s line %d): %r", url_file.name, lineno, line)
                continue

            dedup_key = normalized.rstrip("/")
            if dedup_key in seen:
                logger.warning(
                    "Duplicate URL skipped (%s line %d): %r — already seen as %r",
                    url_file.name, lineno, line, seen[dedup_key],
                )
                continue

            if normalized != line:
                logger.info("URL normalised (%s line %d): %r → %r", url_file.name, lineno, line, normalized)

            seen[dedup_key] = normalized
            urls.append(normalized)

    return urls


def load_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return [
        f for f in DATA_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and f.name not in {p.name for p in URLS_FILES}
    ]


def run(urls: list[str], files: list[Path]) -> None:
    from rag import RagConfig, RagPipeline

    if not urls and not files:
        logger.warning("Nothing to ingest. Add URLs to data/urls.txt or drop files in data/")
        sys.exit(0)

    pipeline = RagPipeline(RagConfig.from_env())
    results = []

    for url in urls:
        logger.info("Ingesting URL: %s", url)
        try:
            r = pipeline.ingest_url(url)
            results.append(r)
            _log_result(r, url)
        except Exception as e:
            logger.error("  ✗ %s — %s", url, e)

    for path in files:
        logger.info("Ingesting file: %s", path.name)
        try:
            r = pipeline.ingest_file(str(path))
            results.append(r)
            _log_result(r, path.name)
        except Exception as e:
            logger.error("  ✗ %s — %s", path.name, e)

    ok = sum(1 for r in results if r.get("status") in ("ok", "skipped"))
    new = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    total_chunks = sum(r.get("chunks", 0) for r in results if r.get("status") == "ok")
    print(
        f"\nDone. {ok}/{len(results)} sources OK "
        f"({new} ingested, {skipped} unchanged) — {total_chunks} new chunks stored."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data into RAG ChromaDB store")
    parser.add_argument("--url", metavar="URL", nargs="*", help="Extra URLs to ingest (in addition to urls.txt / web_urls.txt)")
    parser.add_argument("--file", metavar="FILE", nargs="*", help="Extra files to ingest (in addition to data/ folder)")
    parser.add_argument("--urls-only", action="store_true", help="Skip files, only process URLs")
    parser.add_argument("--files-only", action="store_true", help="Skip URLs, only process files")
    args = parser.parse_args()

    urls = ([] if args.files_only else load_urls()) + (args.url or [])
    files = ([] if args.urls_only else load_files()) + [Path(f) for f in (args.file or [])]

    run(urls, files)


if __name__ == "__main__":
    main()
