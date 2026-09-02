"""
AegisAI — PhishTank Dataset Downloader

Downloads the official PhishTank verified online phishing CSV into data/raw/.

Official source: https://data.phishtank.com/data/online-valid.csv

Usage:
    python scripts/download_phishtank.py
    python scripts/download_phishtank.py --force   # Re-download even if file exists
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import SETTINGS

PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
OUTPUT_FILENAME = "phishtank_online_valid.csv"
USER_AGENT = "phishtank/AegisAI-academic-project"


def download_phishtank(force: bool = False) -> None:
    """Download the PhishTank verified online phishing database.

    Parameters
    ----------
    force : bool
        If True, re-download even if the file already exists.
    """
    import urllib.request
    import urllib.error

    output_path = SETTINGS.DATA_RAW_DIR / OUTPUT_FILENAME
    SETTINGS.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] File already exists: {output_path} ({size_mb:.1f} MB)")
        print("[INFO] Use --force to re-download.")
        return

    print(f"[INFO] Downloading PhishTank database from:")
    print(f"       {PHISHTANK_URL}")
    print(f"[INFO] Saving to: {output_path}")

    request = urllib.request.Request(
        PHISHTANK_URL,
        headers={"User-Agent": USER_AGENT},
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()

                if len(data) == 0:
                    print("[ERROR] Downloaded file is empty. Aborting.")
                    sys.exit(1)

                output_path.write_bytes(data)
                size_mb = len(data) / (1024 * 1024)
                print(f"[INFO] Download complete: {size_mb:.1f} MB")
                return

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * attempt
                print(f"[WARN] Rate limited (HTTP 429). Waiting {wait}s before retry {attempt}/{max_retries}...")
                time.sleep(wait)
            else:
                print(f"[ERROR] HTTP error {e.code}: {e.reason}")
                sys.exit(1)

        except urllib.error.URLError as e:
            print(f"[ERROR] Connection error: {e.reason}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"[INFO] Retrying in {wait}s ({attempt}/{max_retries})...")
                time.sleep(wait)
            else:
                print("[ERROR] All retries exhausted. Aborting.")
                sys.exit(1)

    print("[ERROR] Download failed after all retries.")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the PhishTank verified phishing URL database",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()
    download_phishtank(force=args.force)


if __name__ == "__main__":
    main()
