"""Download checksum-verified Hyser files from PhysioNet v1.0.0.

By default, this downloads preprocessed EMG and force WFDB records from the
1-DoF dataset. Use ``--all`` to download every file listed in the manifest.
"""

import argparse
import hashlib
import re
import sys
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import quote

BASE_URL = "https://physionet.org/files/hd-semg/1.0.0/"
MANIFEST_URL = f"{BASE_URL}SHA256SUMS.txt"
TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_DEFAULT_FILE_RE = re.compile(
    r"^1dof_dataset/subject(?P<subject>0[1-9]|1[0-9]|20)_session"
    r"(?P<session>[12])/1dof_(?:preprocess|force)_[^/]+\.(?:hea|dat)$"
)


@dataclass(frozen=True)
class ManifestEntry:
    """A file path and its expected SHA-256 digest."""

    sha256: str
    path: str


def _validate_manifest_path(path: str) -> PurePosixPath:
    """Return a safe relative POSIX path or raise ``ValueError``."""
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    parts = path.split("/")
    if (
        not path
        or "\x00" in path
        or "\\" in path
        or posix_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe manifest path: {path!r}")
    return posix_path


def parse_manifest(text: str) -> list[ManifestEntry]:
    """Parse ``SHA256SUMS.txt`` into validated manifest entries."""
    entries: list[ManifestEntry] = []
    seen_paths: set[str] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _SHA256_RE.fullmatch(fields[0]):
            raise ValueError(f"invalid manifest entry on line {line_number}")

        checksum, path = fields
        path = path.strip()
        if path.startswith("*"):
            path = path[1:]
        normalized_path = _validate_manifest_path(path).as_posix()
        if normalized_path in seen_paths:
            raise ValueError(f"duplicate manifest path on line {line_number}: {path}")

        seen_paths.add(normalized_path)
        entries.append(ManifestEntry(checksum.lower(), normalized_path))

    if not entries:
        raise ValueError("manifest contains no entries")
    return entries


def fetch_manifest(
    url: str = MANIFEST_URL,
    timeout: float = TIMEOUT_SECONDS,
) -> list[ManifestEntry]:
    """Fetch and parse the PhysioNet checksum manifest."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        contents = response.read()
    return parse_manifest(contents.decode("utf-8"))


def select_entries(
    entries: Iterable[ManifestEntry],
    *,
    all_files: bool = False,
    subjects: Iterable[int] | None = None,
    sessions: Iterable[int] | None = None,
) -> list[ManifestEntry]:
    """Select all files or the requested default 1-DoF records."""
    entries = list(entries)
    if all_files:
        return entries

    subject_filter = set(subjects) if subjects is not None else set(range(1, 21))
    session_filter = set(sessions) if sessions is not None else {1, 2}
    selected: list[ManifestEntry] = []
    for entry in entries:
        match = _DEFAULT_FILE_RE.fullmatch(entry.path)
        if match is None:
            continue
        if (
            int(match.group("subject")) in subject_filter
            and int(match.group("session")) in session_filter
        ):
            selected.append(entry)
    return selected


def _destination_path(dest_root: Path, relative_path: str) -> Path:
    """Map a manifest path below ``dest_root`` without allowing escapes."""
    manifest_path = _validate_manifest_path(relative_path)
    root_resolved = dest_root.resolve(strict=False)
    destination = dest_root.joinpath(*manifest_path.parts)
    try:
        destination.resolve(strict=False).relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes destination: {relative_path!r}") from exc
    return destination


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    entry: ManifestEntry,
    dest_root: Path,
    *,
    dry_run: bool = False,
    timeout: float = TIMEOUT_SECONDS,
) -> bool:
    """Download one entry atomically and return whether it succeeded."""
    partial_path: Path | None = None
    clean_partial = False
    try:
        destination = _destination_path(dest_root, entry.path)
        url = f"{BASE_URL}{quote(entry.path, safe='/')}"

        if destination.exists():
            if _sha256_file(destination) == entry.sha256:
                print(f"  [skip] {entry.path} (checksum verified)")
                return True
            print(f"  [replace] {entry.path} (existing checksum mismatch)")

        if dry_run:
            print(f"  [dry-run] {url}")
            return True

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial_path = _destination_path(dest_root, f"{entry.path}.part")
        clean_partial = True
        digest = hashlib.sha256()

        print(f"  [download] {entry.path}")
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with partial_path.open("wb") as output:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)

        actual_checksum = digest.hexdigest()
        if actual_checksum != entry.sha256:
            raise ValueError(f"checksum mismatch: expected {entry.sha256}, got {actual_checksum}")

        partial_path.replace(destination)
        clean_partial = False
        print(f"  [verified] {entry.path}")
        return True
    except Exception as exc:
        print(f"  [error] {entry.path}: {exc}", file=sys.stderr)
        return False
    finally:
        if clean_partial and partial_path is not None:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"  [warn] could not remove {partial_path}: {exc}", file=sys.stderr)


def download_entries(
    entries: Iterable[ManifestEntry],
    dest_root: Path,
    *,
    dry_run: bool = False,
    timeout: float = TIMEOUT_SECONDS,
) -> int:
    """Download every entry and return the number of failures."""
    failures = 0
    for entry in entries:
        try:
            succeeded = download_file(
                entry,
                dest_root,
                dry_run=dry_run,
                timeout=timeout,
            )
        except Exception as exc:
            print(f"  [error] {entry.path}: {exc}", file=sys.stderr)
            succeeded = False
        if not succeeded:
            failures += 1
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download checksum-verified Hyser files from PhysioNet v1.0.0.",
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
        dest="all_files",
        help="Download every file in the manifest; subject/session filters are ignored.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        type=int,
        choices=range(1, 21),
        metavar="N",
        help="Limit the default selection to subject N; may be repeated.",
    )
    parser.add_argument(
        "--session",
        action="append",
        type=int,
        choices=range(1, 3),
        metavar="N",
        help="Limit the default selection to session N; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing files or directories.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Fetching manifest: {MANIFEST_URL}")
    try:
        manifest = fetch_manifest()
    except Exception as exc:
        print(f"Failed to fetch manifest: {exc}", file=sys.stderr)
        return 1

    entries = select_entries(
        manifest,
        all_files=args.all_files,
        subjects=args.subject,
        sessions=args.session,
    )
    if not entries:
        print("No manifest files matched the requested selection.", file=sys.stderr)
        return 1

    dest_root = Path(args.dest)
    print(f"Selected {len(entries)} files for {dest_root}")
    failures = download_entries(entries, dest_root, dry_run=args.dry_run)
    if failures:
        print(f"Failed to download {failures} file(s).", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Dataset saved to: {dest_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
