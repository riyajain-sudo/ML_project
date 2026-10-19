"""Optional ChEMBL acquisition with bounded pages, fail-closed checks, and provenance."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGET_ID = "CHEMBL2094139"
PAGE_SIZE = 1000
TIMEOUT_SECONDS = 30
OUTPUT = ROOT / "data/gyrB_external_download.csv"
MANIFEST = ROOT / "data/gyrB_external_download_manifest.json"

def fetch_activities(session=requests, target_id=TARGET_ID, page_size=PAGE_SIZE) -> tuple[list[dict], int]:
    rows, offset, pages, total = [], 0, 0, None
    seen_offsets = set()
    while total is None or offset < total:
        if offset in seen_offsets:
            raise RuntimeError(f"ChEMBL pagination repeated offset {offset}; aborting")
        seen_offsets.add(offset)
        response = session.get(URL, params={"target_chembl_id": target_id, "limit": page_size, "offset": offset}, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        try:
            payload = response.json()
            page = payload["activities"]
            meta = payload["page_meta"]
            page_total = int(meta["total_count"])
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError("ChEMBL response lacks valid activities/page_meta.total_count") from exc
        if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
            raise RuntimeError("ChEMBL activities must be a list of objects")
        if total is None:
            total = page_total
        elif page_total != total:
            raise RuntimeError("ChEMBL total_count changed during pagination; refusing partial snapshot")
        pages += 1
        rows.extend(page)
        if not page and offset < total:
            raise RuntimeError("ChEMBL returned an empty page before total_count was reached")
        offset += len(page)
        if len(page) > page_size:
            raise RuntimeError("ChEMBL returned more rows than requested page size")
    if len(rows) != total:
        raise RuntimeError(f"Fetched {len(rows)} activities but API reports {total}")
    return rows, pages

def main() -> None:
    try:
        rows, pages = fetch_activities()
        frame = pd.DataFrame(rows)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        csv = frame.to_csv(index=False, lineterminator="\n")
        OUTPUT.write_text(csv, encoding="utf-8", newline="")
        checksum = hashlib.sha256(csv.encode("utf-8")).hexdigest()
        manifest = {"source":"ChEMBL REST activity API", "url":URL, "target_chembl_id":TARGET_ID,
                    "retrieved_at_utc":datetime.now(timezone.utc).isoformat(), "page_size":PAGE_SIZE,
                    "page_count":pages, "row_count":len(rows), "sha256_csv":checksum,
                    "query":"target_chembl_id=CHEMBL2094139; all activity types as returned"}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(rows)} rows from {pages} pages to {OUTPUT}; SHA-256 {checksum}")
    except (requests.RequestException, RuntimeError, OSError) as exc:
        print(f"ChEMBL download failed; no partial export was accepted: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
