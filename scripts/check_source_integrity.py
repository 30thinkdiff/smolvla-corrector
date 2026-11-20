"""Verify committed file bytes against a pinned provenance manifest.

Standard library only; this runs without network, GPU or model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()


def verify(root: Path = ROOT) -> int:
    info = json.loads((root / "provenance/source_map.json").read_text(encoding="utf-8"))
    count = 0
    for relative, metadata in sorted(info["files"].items()):
        path = root / relative
        if ".." in Path(relative).parts or not path.is_file():
            raise ValueError(f"Missing or invalid source path: {relative}")
        actual = blob_sha(path.read_bytes())
        expected = metadata.get("migrated_git_blob_sha", metadata["source_git_blob_sha"])
        if actual != expected:
            raise ValueError(f"Committed file digest mismatch: {relative}")
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(f"PASS: {verify(args.root)} committed Git blobs verified")
