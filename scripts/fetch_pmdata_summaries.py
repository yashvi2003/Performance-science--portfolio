"""Fetch daily wellness, sRPE training logs, and sleep scores from OSF PMData repository.

Downloads only the daily summary CSVs (~300 KB total across all 16 participants):
- /pmdata/{p}/pmsys/wellness.csv
- /pmdata/{p}/pmsys/srpe.csv
- /pmdata/{p}/fitbit/sleep_score.csv

Avoids downloading the massive ~1.4 GB raw minute-level heart rate JSONs and food images.
"""
import json
import logging
from pathlib import Path
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_pmdata")

BASE_API = "https://api.osf.io/v2"
PMDATA_FOLDER_ID = "5e99d05ef135350590d5316d"
RAW_DIR = Path("data/raw/pmdata")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "NeuromuscularMonitoringToolkit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "NeuromuscularMonitoringToolkit/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()
    dest_path.write_bytes(content)
    logger.info("  Saved: %s (%d bytes)", dest_path, len(content))


def get_all_participants() -> dict[str, str]:
    """Retrieve mapping of participant name (p01..p16) to OSF folder ID."""
    participants = {}
    url = f"{BASE_API}/nodes/vx4bk/files/osfstorage/{PMDATA_FOLDER_ID}/"
    while url:
        data = fetch_json(url)
        for item in data.get("data", []):
            name = item["attributes"]["name"]
            if name.startswith("p") and item["attributes"]["kind"] == "folder":
                participants[name] = item["id"]
        url = data.get("links", {}).get("next")
    return dict(sorted(participants.items()))


def get_subfolders(folder_id: str) -> dict[str, str]:
    """Get mapping of subfolder name -> folder ID (e.g. 'fitbit', 'pmsys')."""
    subfolders = {}
    url = f"{BASE_API}/nodes/vx4bk/files/osfstorage/{folder_id}/"
    data = fetch_json(url)
    for item in data.get("data", []):
        if item["attributes"]["kind"] == "folder":
            subfolders[item["attributes"]["name"]] = item["id"]
    return subfolders


def get_folder_files(folder_id: str) -> dict[str, str]:
    """Get mapping of filename -> download URL."""
    files = {}
    url = f"{BASE_API}/nodes/vx4bk/files/osfstorage/{folder_id}/"
    while url:
        data = fetch_json(url)
        for item in data.get("data", []):
            if item["attributes"]["kind"] == "file":
                fname = item["attributes"]["name"]
                dl_url = item["links"]["download"]
                files[fname] = dl_url
        url = data.get("links", {}).get("next")
    return files


def fetch_all_pmdata_summaries() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Discovering participants in PMData repository...")
    participants = get_all_participants()
    logger.info("Found %d participants: %s", len(participants), list(participants.keys()))

    target_files = {
        "pmsys": ["wellness.csv", "srpe.csv", "injury.csv"],
        "fitbit": ["sleep_score.csv", "time_in_heart_rate_zones.json"],
    }

    for p_name, p_folder_id in participants.items():
        p_dir = RAW_DIR / p_name
        logger.info("Processing participant %s...", p_name)
        subfolders = get_subfolders(p_folder_id)

        for domain, files_needed in target_files.items():
            if domain not in subfolders:
                logger.warning("  Domain '%s' not found for %s", domain, p_name)
                continue

            folder_files = get_folder_files(subfolders[domain])
            for target_name in files_needed:
                dest = p_dir / domain / target_name
                if dest.exists() and dest.stat().st_size > 0:
                    logger.debug("  Already exists: %s", dest)
                    continue

                if target_name in folder_files:
                    dl_url = folder_files[target_name]
                    try:
                        download_file(dl_url, dest)
                        time.sleep(0.2)  # Respectful rate limiting
                    except Exception as e:
                        logger.error("  Failed to download %s for %s: %s", target_name, p_name, e)
                else:
                    logger.warning("  Target file %s not found in %s/%s", target_name, p_name, domain)

    logger.info("PMData summary download completed.")


if __name__ == "__main__":
    fetch_all_pmdata_summaries()
