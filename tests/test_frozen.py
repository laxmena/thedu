from thedu.cli import MANIFEST, frozen_digests, read_manifest
from thedu.paths import ROOT


def test_frozen_files_are_unchanged():
    assert MANIFEST.exists(), "run `thedu freeze` once"
    stored = read_manifest()
    import hashlib

    for path, digest in stored.items():
        f = ROOT / path
        assert f.exists(), f"frozen file {path} vanished"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == digest, f"frozen file {path} changed"
        assert not f.stat().st_mode & 0o222, f"{path} is writable"


def test_every_judge_qrels_and_golden_file_is_frozen():
    # A new file in a frozen folder must be added with `thedu freeze --add`, not slip in unchecked.
    assert set(frozen_digests()) <= set(read_manifest())
