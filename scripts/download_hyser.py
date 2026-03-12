"""Download the Hyser HD-sEMG dataset from PhysioNet.

Usage
-----
    python scripts/download_hyser.py --dest data/hyser

PhysioNet credentials are required. Set them via environment variables
PHYSIONET_USER and PHYSIONET_PASSWORD, or pass them on the command line.

Dataset
-------
Hyser: A High-Density Surface EMG Dataset
https://physionet.org/content/hd-semg/1.0.0/
"""

import argparse
import os
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


def make_opener(username: str, password: str) -> urllib.request.OpenerDirector:
    """Create a URL opener with HTTP Basic Auth."""
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, BASE_URL, username, password)
    auth_handler = urllib.request.HTTPBasicAuthHandler(manager)
    return urllib.request.build_opener(auth_handler)


def download_file(
    opener: urllib.request.OpenerDirector,
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
    print(f"  Downloading {url} → {dest}")
    try:
        with opener.open(url) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
    except urllib.error.HTTPError as exc:
        print(f"  [warn] HTTP {exc.code} for {url}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Hyser HD-sEMG dataset from PhysioNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        default="data/hyser",
        help="Local directory to save the dataset.",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("PHYSIONET_USER", ""),
        help="PhysioNet username (or set PHYSIONET_USER env var).",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("PHYSIONET_PASSWORD", ""),
        help="PhysioNet password (or set PHYSIONET_PASSWORD env var).",
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

    if not args.user or not args.password:
        parser.error(
            "PhysioNet credentials are required.\n"
            "Set PHYSIONET_USER and PHYSIONET_PASSWORD environment variables,\n"
            "or pass --user and --password on the command line.\n"
            "Register at https://physionet.org/register/ if you don't have an account."
        )

    dest_root = Path(args.dest)
    files = build_file_list(full=args.full)
    opener = make_opener(args.user, args.password)

    print(f"Downloading {len(files)} files to {dest_root}/")
    for rel_path in files:
        url = urljoin(BASE_URL, rel_path)
        local_path = dest_root / rel_path
        download_file(opener, url, local_path, dry_run=args.dry_run)

    print("Done.")
    if not args.dry_run:
        print(f"Dataset saved to: {dest_root.resolve()}")


if __name__ == "__main__":
    main()
