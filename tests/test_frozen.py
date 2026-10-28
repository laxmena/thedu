from thedu.cli import MANIFEST, frozen_digests
from thedu.paths import ROOT


def test_frozen_files_are_unchanged():
    assert MANIFEST.exists(), "run `thedu freeze` once at the end of M0"
    stored = {}
    for ln in MANIFEST.read_text().splitlines():
        digest, path = ln.split("  ", 1)
        stored[path] = digest
    assert frozen_digests() == stored, "a frozen file (judge/, data/qrels/, tests/golden/) changed, appeared or vanished"
    for path in stored:
        assert not (ROOT / path).stat().st_mode & 0o222, f"{path} is writable"
