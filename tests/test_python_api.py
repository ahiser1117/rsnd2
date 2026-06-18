from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

import rsnd2 as nd2  # noqa: E402
from rsnd2 import _ffi  # noqa: E402


MAGIC = bytes([0xDA, 0xCE, 0xBE, 0x0A])
FOOTER = b"ND2 CHUNK MAP SIGNATURE 0000001!"


def chunk(name: str, payload: bytes) -> bytes:
    name_bytes = name.encode() + b"\0"
    return MAGIC + struct.pack("<IQ", len(name_bytes), len(payload)) + name_bytes + payload


def fixture(path: Path) -> None:
    out = bytearray()
    entries: list[tuple[str, int]] = []

    out.extend(chunk("ND2 FILE SIGNATURE CHUNK NAME01!", b"Ver3.0\0"))
    for sequence in range(2):
        entries.append((f"ImageDataSeq|{sequence}!", len(out)))
        pixels = struct.pack("<4H", sequence, sequence + 1, sequence + 2, sequence + 3)
        out.extend(chunk(f"ImageDataSeq|{sequence}!", b"\0" * 8 + pixels))

    filemap_offset = len(out)
    payload = bytearray()
    for name, offset in entries:
        payload.extend(name.encode())
        payload.extend(struct.pack("<QQ", offset, 0))
    out.extend(chunk("ND2 FILEMAP SIGNATURE NAME0001!", bytes(payload)))
    out.extend(FOOTER)
    out.extend(struct.pack("<Q", filemap_offset))
    path.write_bytes(out)


class TestPythonApi(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "sample.nd2"
        fixture(self.path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_top_level_probe_api(self) -> None:
        self.assertTrue(nd2.is_supported_file(self.path))
        self.assertFalse(nd2.is_legacy(self.path))
        self.assertTrue(nd2.ND2File.is_supported_file(self.path))

    def test_index_and_metadata_shape(self) -> None:
        with nd2.ND2File(self.path) as f:
            self.assertEqual(f.path, str(self.path))
            self.assertFalse(f.closed)
            self.assertEqual(f.version, (3, 0))
            self.assertEqual(f.attributes.sequenceCount, 2)
            self.assertEqual(dict(f.sizes), {"T": 2, "Y": 2, "X": 2})
            self.assertEqual(f.shape, (2, 2, 2))
            self.assertEqual(f.metadata.contents.frameCount, 2)
            self.assertEqual(f.events(), [])
        self.assertTrue(f.closed)

    def test_payload_api(self) -> None:
        payload = _ffi.read_plane_payload(self.path, 1)
        self.assertEqual(payload[:8], b"\0" * 8)
        self.assertEqual(struct.unpack("<4H", payload[8:]), (1, 2, 3, 4))

    @unittest.skipIf(importlib.util.find_spec("numpy") is None, "NumPy is not installed")
    def test_array_api_when_numpy_is_available(self) -> None:
        with nd2.ND2File(self.path) as f:
            arr = f.asarray()
            self.assertEqual(arr.shape, (2, 2, 2))
            self.assertEqual(arr.dtype.name, "uint16")
            self.assertEqual(arr[1, 1, 1], 4)


if __name__ == "__main__":
    unittest.main()
