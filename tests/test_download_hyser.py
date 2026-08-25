"""Tests for the manifest-driven Hyser downloader."""

import hashlib
import io

import pytest

from scripts import download_hyser


def _entry(path: str, contents: bytes = b"contents") -> download_hyser.ManifestEntry:
    return download_hyser.ManifestEntry(hashlib.sha256(contents).hexdigest(), path)


class _FakeResponse(io.BytesIO):
    def __init__(self, contents: bytes, on_read=None):
        super().__init__(contents)
        self._on_read = on_read

    def read(self, size: int = -1) -> bytes:
        if self._on_read is not None:
            self._on_read()
        return super().read(size)


def test_parse_manifest_and_reject_unsafe_paths():
    checksum_a = "A" * 64
    checksum_b = "b" * 64
    manifest = (
        f"{checksum_a}  1dof_dataset/subject01_session1/file.dat\n"
        f"{checksum_b} *folder/file with spaces.hea\n"
    )

    assert download_hyser.parse_manifest(manifest) == [
        download_hyser.ManifestEntry(
            checksum_a.lower(), "1dof_dataset/subject01_session1/file.dat"
        ),
        download_hyser.ManifestEntry(checksum_b, "folder/file with spaces.hea"),
    ]

    with pytest.raises(ValueError, match="unsafe manifest path"):
        download_hyser.parse_manifest(f"{checksum_b}  ../outside.dat\n")
    with pytest.raises(ValueError, match="invalid manifest entry"):
        download_hyser.parse_manifest("not-a-checksum file.dat\n")


def test_select_entries_defaults_filters_and_all():
    default_subject_1 = _entry(
        "1dof_dataset/subject01_session1/1dof_preprocess_finger1_sample1.dat"
    )
    default_force = _entry("1dof_dataset/subject01_session2/1dof_force_finger1_sample1.hea")
    default_subject_2 = _entry(
        "1dof_dataset/subject02_session2/1dof_preprocess_finger1_sample1.hea"
    )
    raw = _entry("1dof_dataset/subject01_session1/1dof_raw_finger1_sample1.dat")
    other_dataset = _entry("mvc_dataset/subject01_session1/mvc_force_finger1_flexion.dat")
    entries = [default_subject_1, default_force, default_subject_2, raw, other_dataset]

    assert download_hyser.select_entries(entries) == [
        default_subject_1,
        default_force,
        default_subject_2,
    ]
    assert download_hyser.select_entries(entries, subjects=[1], sessions=[2]) == [default_force]
    assert (
        download_hyser.select_entries(
            entries,
            all_files=True,
            subjects=[1],
            sessions=[1],
        )
        == entries
    )


def test_fetch_manifest_uses_timeout(monkeypatch):
    checksum = "1" * 64
    response = _FakeResponse(f"{checksum} file.dat\n".encode())
    calls = []

    def fake_urlopen(url, *, timeout):
        calls.append((url, timeout))
        return response

    monkeypatch.setattr(download_hyser.urllib.request, "urlopen", fake_urlopen)

    assert download_hyser.fetch_manifest(timeout=7) == [
        download_hyser.ManifestEntry(checksum, "file.dat")
    ]
    assert calls == [(download_hyser.MANIFEST_URL, 7)]


def test_dry_run_does_not_use_network_or_create_directories(tmp_path, monkeypatch):
    entry = _entry("1dof_dataset/subject01_session1/1dof_force_finger1_sample1.dat")
    destination = tmp_path / "missing" / "destination"

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("dry-run must not download files")

    monkeypatch.setattr(download_hyser.urllib.request, "urlopen", fail_urlopen)

    assert download_hyser.download_entries([entry], destination, dry_run=True) == 0
    assert not destination.exists()


def test_download_replaces_corrupt_file_only_after_verified_stream(tmp_path, monkeypatch):
    contents = b"verified downloaded contents"
    entry = _entry(
        "1dof_dataset/subject01_session1/1dof_preprocess_finger1_sample1.dat",
        contents,
    )
    destination = tmp_path / entry.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old corrupt contents")
    calls = []

    def assert_original_is_still_present():
        assert destination.read_bytes() == b"old corrupt contents"

    def fake_urlopen(url, *, timeout):
        calls.append((url, timeout))
        return _FakeResponse(contents, on_read=assert_original_is_still_present)

    monkeypatch.setattr(download_hyser.urllib.request, "urlopen", fake_urlopen)

    assert download_hyser.download_file(entry, tmp_path)
    assert destination.read_bytes() == contents
    assert not destination.with_name(f"{destination.name}.part").exists()
    assert calls == [(f"{download_hyser.BASE_URL}{entry.path}", download_hyser.TIMEOUT_SECONDS)]


def test_verified_existing_file_is_skipped(tmp_path, monkeypatch):
    contents = b"already complete"
    entry = _entry("records/example.dat", contents)
    destination = tmp_path / entry.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(contents)

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("verified files must not be downloaded again")

    monkeypatch.setattr(download_hyser.urllib.request, "urlopen", fail_urlopen)

    assert download_hyser.download_file(entry, tmp_path)
    assert destination.read_bytes() == contents


def test_checksum_failure_removes_partial_and_preserves_existing_file(tmp_path, monkeypatch):
    entry = _entry("records/example.dat", b"expected contents")
    destination = tmp_path / entry.path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing corrupt contents")

    monkeypatch.setattr(
        download_hyser.urllib.request,
        "urlopen",
        lambda url, *, timeout: _FakeResponse(b"wrong downloaded contents"),
    )

    assert not download_hyser.download_file(entry, tmp_path)
    assert destination.read_bytes() == b"existing corrupt contents"
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_main_aggregates_failures_without_network(tmp_path, monkeypatch):
    entries = [
        _entry("1dof_dataset/subject01_session1/1dof_force_finger1_sample1.dat"),
        _entry("1dof_dataset/subject01_session1/1dof_force_finger1_sample1.hea"),
    ]
    attempted = []

    monkeypatch.setattr(download_hyser, "fetch_manifest", lambda: entries)

    def fake_download(entry, dest_root, *, dry_run, timeout):
        attempted.append((entry, dest_root, dry_run, timeout))
        return entry is entries[1]

    monkeypatch.setattr(download_hyser, "download_file", fake_download)

    result = download_hyser.main(
        ["--dest", str(tmp_path / "download"), "--subject", "1", "--session", "1"]
    )

    assert result == 1
    assert [call[0] for call in attempted] == entries
