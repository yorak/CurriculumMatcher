"""
Fetch ITC/Computing Sciences course units from the Tampere University Kori API
and embed them into a local ChromaDB vector store.

Run once before starting the app:
    python fetch_and_index.py

Options:
    --limit N     Stop after indexing N courses (gentle on the API, good for testing)
    --explore     Print a raw API sample response and exit
"""

import argparse
import html
import json
import os
import re
import sys
import time

import chromadb
import requests

KORI_BASE = "https://sis-tuni.funidata.fi/kori/api"
UNIVERSITY_ID = "tuni-university-root-id"
CURRICULUM_PERIOD = "uta-lvv-2024"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "curriculum_itc"

# Code prefixes confirmed to have courses in uta-lvv-2024
ITC_PREFIXES = ("COMP", "DATA", "TIE")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loc(field, fallback="") -> str:
    """Extract best available language from a localized string dict."""
    if not field:
        return fallback
    if isinstance(field, str):
        return field
    return field.get("en") or field.get("fi") or field.get("sv") or fallback


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r" +", " ", text).strip()


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["Accept"] = "application/json"
    return s


def get_with_retry(session, url, params, retries=3) -> dict:
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"  ERROR: {exc}", file=sys.stderr)
                return {}
            wait = 2 ** attempt
            print(f"  Retry in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return {}


# ---------------------------------------------------------------------------
# Kori API fetching (two-step)
# ---------------------------------------------------------------------------

def fetch_groupids(session, prefix: str, max_count: int = None) -> list:
    """
    Step 1: search by code prefix to collect groupIds.
    Stops early if max_count is reached (gentle on the API).
    """
    groupids = []
    start = 0
    page = 100

    while True:
        fetch_n = min(page, max_count - len(groupids)) if max_count else page

        data = get_with_retry(session, f"{KORI_BASE}/course-unit-search", {
            "universityId": UNIVERSITY_ID,
            "curriculumPeriodId": CURRICULUM_PERIOD,
            "codeQuery": prefix,
            "start": start,
            "limit": fetch_n,
        })

        batch = data.get("searchResults", [])
        for item in batch:
            gid = item.get("groupId")
            if gid:
                groupids.append(gid)

        done = (not batch
                or len(batch) < fetch_n
                or (max_count and len(groupids) >= max_count))
        if done:
            break
        start += fetch_n

    return groupids


def fetch_detail(session, groupid: str) -> dict:
    """
    Step 2: fetch full course data (content + outcomes) by groupId.
    Returns the first (most current) version.
    """
    data = get_with_retry(session, f"{KORI_BASE}/course-units/by-group-id", {
        "groupId": groupid,
        "universityId": UNIVERSITY_ID,
    })
    if isinstance(data, list) and data:
        return data[0]
    return {}


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def build_doc(course: dict) -> str:
    code = course.get("code", "")
    name = loc(course.get("name"))
    creds = course.get("credits") or {}
    cmin = safe_float(creds.get("min"))
    cmax = safe_float(creds.get("max"))
    credit_str = f"{cmin:.0f}–{cmax:.0f}" if (cmin or cmax) else "?"

    content = strip_html(
        loc(course.get("content"))
        or loc(course.get("learningMaterial"))
        or loc(course.get("additional"))
    )
    outcomes = strip_html(loc(course.get("outcomes")))

    parts = [f"Course: {code} - {name}", f"Credits: {credit_str} ECTS"]
    if content:
        parts.append(f"Content: {content}")
    if outcomes:
        parts.append(f"Learning Outcomes: {outcomes}")

    return "\n".join(parts)


def build_metadata(course: dict) -> dict:
    creds = course.get("credits") or {}
    name = loc(course.get("name"), course.get("code", ""))
    return {
        "code": course.get("code", ""),
        "name": name[:200],
        "credits_min": safe_float(creds.get("min")),
        "credits_max": safe_float(creds.get("max")),
    }


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_courses(courses: list) -> int:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    docs, ids, metas = [], [], []
    seen_ids = set()

    for i, course in enumerate(courses):
        doc = build_doc(course)
        if len(doc.strip()) < 40:
            continue

        raw_id = course.get("id") or course.get("groupId") or f"course-{i}"
        uid = raw_id
        suffix = 0
        while uid in seen_ids:
            suffix += 1
            uid = f"{raw_id}-{suffix}"
        seen_ids.add(uid)

        docs.append(doc)
        ids.append(uid)
        metas.append(build_metadata(course))

    for start in range(0, len(docs), 100):
        collection.add(
            documents=docs[start:start + 100],
            ids=ids[start:start + 100],
            metadatas=metas[start:start + 100],
        )

    return len(docs)


# ---------------------------------------------------------------------------
# Explore mode
# ---------------------------------------------------------------------------

def explore():
    """Print a sample raw API response to help verify field names."""
    session = make_session()
    data = get_with_retry(session, f"{KORI_BASE}/course-unit-search", {
        "universityId": UNIVERSITY_ID,
        "curriculumPeriodId": CURRICULUM_PERIOD,
        "codeQuery": "COMP",
        "start": 0,
        "limit": 2,
    })
    print("=== course-unit-search sample ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    results = data.get("searchResults", [])
    if results:
        gid = results[0].get("groupId")
        detail_data = get_with_retry(session, f"{KORI_BASE}/course-units/by-group-id", {
            "groupId": gid,
            "universityId": UNIVERSITY_ID,
        })
        print(f"\n=== full detail for groupId={gid} ===")
        detail = detail_data[0] if isinstance(detail_data, list) and detail_data else detail_data
        print(json.dumps(detail, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch and index Tampere ITC courses")
    parser.add_argument("--explore", action="store_true",
                        help="Print raw API sample and exit")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Stop after N courses total (e.g. --limit 50 for a quick test)")
    args = parser.parse_args()

    if args.explore:
        explore()
        return

    session = make_session()
    all_details = []

    for prefix in ITC_PREFIXES:
        remaining = (args.limit - len(all_details)) if args.limit else None
        if args.limit and remaining <= 0:
            break

        print(f"Fetching {prefix} course IDs...")
        groupids = fetch_groupids(session, prefix, max_count=remaining)
        print(f"  Found {len(groupids)} groupIds, fetching full details...")

        for i, gid in enumerate(groupids):
            detail = fetch_detail(session, gid)
            if detail:
                all_details.append(detail)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(groupids)} details fetched")

    print(f"\nTotal courses fetched: {len(all_details)}")

    if not all_details:
        print("No courses found. Try --explore to inspect the raw API response.", file=sys.stderr)
        sys.exit(1)

    print("Embedding and indexing into ChromaDB...")
    n = index_courses(all_details)
    print(f"Done. Indexed {n} courses into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
