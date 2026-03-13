"""Download the Hyser HD-sEMG dataset from PhysioNet.

Usage
-----
    python scripts/download_hyser.py --dest data/hyser

No credentials required — the Hyser dataset is open access.

Dataset
-------
Hyser: A High-Density Surface EMG Dataset
https://physionet.org/content/hd-semg/1.0.0/
"""

import argparse
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "https://physionet.org/files/hd-semg/1.0.0/"

# Subset of files to download by default (1-DOF protocol, subjects 1-20).
# Pass --all to fetch the full dataset (~35 GB).
DEFAULT_FILES = [
    "RECORDS",
    "SHA256SUMS.txt",
]

# 1-DOF session files for all 20 subjects
_1DOF_TEMPLATE = "1dof/subject{subj:02d}/session{sess}.mat"
_1DOF_SUBJECTS = range(1, 21)
_1DOF_SESSIONS = range(1, 6)


def build_file_list(full: bool) -> list[str]:
    """Return the list of relative file paths to download."""
    files = list(DEFAULT_FILES)
    for subj in _1DOF_SUBJECTS:
        for sess in _1DOF_SESSIONS:
            files.append(_1DOF_TEMPLATE.format(subj=subj, sess=sess))
    if full:
        # Add gesture and rest protocol files as well
        for subj in _1DOF_SUBJECTS:
            files.append(f"gesture/subject{subj:02d}/session1.mat")
    return files


def _format_size(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def download_file(
    url: str,
    dest: Path,
    dry_run: bool = False,
) -> None:
    """Download a single file to *dest*, skipping if already present."""
    if dest.exists():
        print(f"  [skip] {dest.name} (already downloaded)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [dry-run] would download {url}")
        return
    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB chunks
            label = dest.name
            if total:
                print(f"  {label} ({_format_size(total)})")
            else:
                print(f"  {label}")
            with open(dest, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r    {_format_size(downloaded)} / "
                            f"{_format_size(total)} ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
            if total:
                print()  # newline after progress
    except urllib.error.HTTPError as exc:
        print(f"  [warn] HTTP {exc.code} for {url}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Hyser HD-sEMG dataset from PhysioNet (open access).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        default="data/hyser",
        help="Local directory to save the dataset.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="full",
        help="Download the full dataset including gesture protocol files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    dest_root = Path(args.dest)
    files = build_file_list(full=args.full)

    print(f"Downloading {len(files)} files to {dest_root}/")
    for rel_path in files:
        url = urljoin(BASE_URL, rel_path)
        local_path = dest_root / rel_path
        download_file(url, local_path, dry_run=args.dry_run)

    print("Done.")
    if not args.dry_run:
        print(f"Dataset saved to: {dest_root.resolve()}")


if __name__ == "__main__":
    main()
