from __future__ import annotations

import argparse
import os
import json
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def _ensure_dirs(cfg_root: Path) -> None:
    (cfg_root / "ml" / "data" / "raw").mkdir(parents=True, exist_ok=True)


def download_and_extract(dataset_slug: str, out_dir: Path) -> Path:
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    # If env vars are provided, prefer them over any global kaggle.json.
    # This is more reliable on Windows and avoids leaking secrets into code.
    if kaggle_username and kaggle_key:
        cfg_dir = out_dir.parent.parent / ".kaggle"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "kaggle.json").write_text(
            json.dumps({"username": kaggle_username, "key": kaggle_key}),
            encoding="utf-8",
        )
        os.environ["KAGGLE_CONFIG_DIR"] = str(cfg_dir)

    api = KaggleApi()
    api.authenticate()

    out_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset_slug, path=str(out_dir), quiet=False, unzip=False)

    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No zip file downloaded into: {out_dir}")

    zip_path = zips[0]
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))

    return extract_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imrankhan77/autistic-children-facial-data-set")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    _ensure_dirs(project_root)

    raw_dir = project_root / "ml" / "data" / "raw"

    if os.environ.get("KAGGLE_CONFIG_DIR"):
        print(f"Using KAGGLE_CONFIG_DIR={os.environ['KAGGLE_CONFIG_DIR']}")

    extracted = download_and_extract(args.dataset, raw_dir)
    print(f"Extracted to: {extracted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
