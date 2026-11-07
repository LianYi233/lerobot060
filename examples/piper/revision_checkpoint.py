#!/usr/bin/env python3
"""Lossless, bounded-memory checkpoint bundles sharing one exact frozen base.

Only Python's standard library is needed. A bundle is not directly deployable:
``materialize`` reconstructs the original model and all saved processor files.
These functions never delete source checkpoints or modify the shared base.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

FORMAT = "piper-revision-checkpoint-v1"
MODEL = "model.safetensors"
HEADER = "model.header.bin"
DELTA = "tensors.delta.bin"
MANIFEST = "manifest.json"
BLOCK_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 100_000_000
REQUIRED_SIDECARS = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_loads(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def _reject_symlinks(path):
    path = Path(path).expanduser().absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlinks are not supported: {part}")
    return path


def _directory(path):
    path = _reject_symlinks(path)
    if not path.is_dir():
        raise ValueError(f"Missing directory: {path}")
    return path.resolve()


def _regular_file(path):
    path = _reject_symlinks(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    return path


def _fingerprint(path):
    value = _regular_file(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _unchanged(path, fingerprint):
    if _fingerprint(path) != fingerprint:
        raise ValueError(f"File changed while processing: {path}")


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe bundle path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe bundle path: {name!r}")
    return Path(*path.parts)


def _inventory(root):
    files, directories = {}, []
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        for name in dirs:
            path = _reject_symlinks(Path(current) / name)
            directories.append(path.relative_to(root).as_posix())
        for name in sorted(names):
            path = _regular_file(Path(current) / name)
            files[path.relative_to(root).as_posix()] = _fingerprint(path)
    return files, directories


def _region(stream, offset, length):
    stream.seek(offset)
    remaining = length
    while remaining:
        data = stream.read(min(BLOCK_BYTES, remaining))
        if not data:
            raise ValueError("Unexpected end of tensor data")
        remaining -= len(data)
        yield data


def _sha256(path):
    digest = hashlib.sha256()
    with _regular_file(path).open("rb") as stream:
        for data in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(data)
    return digest.hexdigest()


def _parse_header(prefix, total_bytes):
    if len(prefix) < 8:
        raise ValueError("Truncated safetensors header")
    header_bytes = struct.unpack("<Q", prefix[:8])[0]
    if not 2 <= header_bytes <= MAX_HEADER_BYTES or len(prefix) != 8 + header_bytes:
        raise ValueError("Invalid safetensors header length")
    if type(total_bytes) is not int or total_bytes < len(prefix):
        raise ValueError("Invalid safetensors model size")
    metadata = _json_loads(prefix[8:])
    if not isinstance(metadata, dict):
        raise ValueError("Safetensors header must be a JSON object")
    tensors = {}
    for name, tensor in metadata.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                raise ValueError("Invalid safetensors metadata")
            continue
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor metadata: {name}")
        shape, offsets, dtype = tensor["shape"], tensor["data_offsets"], tensor["dtype"]
        if (
            not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or not isinstance(dtype, str)
            or dtype not in DTYPE_BYTES
        ):
            raise ValueError(f"Unsupported or malformed tensor: {name}")
        start, end = offsets
        if end < start or end > total_bytes - len(prefix):
            raise ValueError(f"Tensor outside payload: {name}")
        if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor shape/dtype does not match its byte length: {name}")
        tensors[name] = tensor
    cursor = 0
    for name in sorted(tensors, key=lambda key: (*tensors[key]["data_offsets"], key)):
        start, end = tensors[name]["data_offsets"]
        if start != cursor:
            raise ValueError(f"Tensor payload has a gap or overlap: {name}")
        cursor = end
    if cursor != total_bytes - len(prefix):
        raise ValueError("Unaccounted bytes after safetensors tensor data")
    return {"prefix": prefix, "tensors": tensors, "size": total_bytes}


def _model_header(path):
    path = _regular_file(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError(f"Invalid safetensors header length: {path}")
        return _parse_header(length_bytes + stream.read(length), size)


def _ordered_tensors(info):
    return sorted(info["tensors"], key=lambda key: (*info["tensors"][key]["data_offsets"], key))


def _base_key_candidates(name):
    # Official PI05 and LeRobot exports can differ by this wrapper prefix.
    # The alias is usable only after verifying dtype, shape and every tensor byte.
    return (name, name.removeprefix("model.")) if name.startswith("model.") else (name, f"model.{name}")


def _copy_hashed(source, destination):
    before = _fingerprint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for data in iter(lambda: incoming.read(BLOCK_BYTES), b""):
            outgoing.write(data)
            digest.update(data)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    _unchanged(source, before)
    return {"bytes": before[2], "sha256": digest.hexdigest()}


@contextmanager
def _output_directory(destination, protected_roots):
    destination = _reject_symlinks(destination)
    if os.path.lexists(destination):
        raise ValueError(f"Output already exists; refusing to overwrite: {destination}")
    destination = destination.resolve()
    if any(destination == root or root in destination.parents for root in protected_roots):
        raise ValueError("Output must be outside the input checkpoint, bundle and base directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinks(destination.parent)
    # Serialize writers using this utility. Never take over a stale lock automatically.
    lock = destination.with_name(f".{destination.name}.revision-write-lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temporary = None
    try:
        os.close(lock_fd)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
        yield temporary
        if os.path.lexists(destination):
            raise ValueError(f"Output appeared while processing: {destination}")
        temporary.rename(destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink()


def _check_space(directory, needed):
    available = shutil.disk_usage(directory).free
    if available < needed:
        raise ValueError(f"Insufficient disk space: need {needed} bytes, available {available} bytes")


def _build_plan(source_path, source, base_path, base):
    segments, delta_offset = [], 0
    source_digest = hashlib.sha256(source["prefix"])
    with source_path.open("rb") as incoming, base_path.open("rb") as original:
        for name in _ordered_tensors(source):
            tensor = source["tensors"][name]
            start, end = tensor["data_offsets"]
            base_key = next((key for key in _base_key_candidates(name) if key in base["tensors"]), None)
            candidate = base["tensors"].get(base_key)
            same = candidate is not None and all(
                candidate[key] == tensor[key] for key in ("dtype", "shape")
            )
            if same:
                original.seek(len(base["prefix"]) + candidate["data_offsets"][0])
            tensor_digest = hashlib.sha256()
            for data in _region(incoming, len(source["prefix"]) + start, end - start):
                source_digest.update(data)
                tensor_digest.update(data)
                if same and original.read(len(data)) != data:
                    same = False
            segments.append({
                "name": name,
                "source": "base" if same else "delta",
                "base_key": base_key if same else None,
                "offset": len(base["prefix"]) + candidate["data_offsets"][0] if same else delta_offset,
                "bytes": end - start,
                "sha256": tensor_digest.hexdigest(),
            })
            if not same:
                delta_offset += end - start
    return segments, source_digest.hexdigest(), delta_offset


def _reconstruct(source, segments, base_path, delta_path, output=None):
    digest = hashlib.sha256(source["prefix"])
    if output is not None:
        output.write(source["prefix"])
    with base_path.open("rb") as original, delta_path.open("rb") as changed:
        for segment in segments:
            stream = original if segment["source"] == "base" else changed
            tensor_digest = hashlib.sha256()
            for data in _region(stream, segment["offset"], segment["bytes"]):
                digest.update(data)
                tensor_digest.update(data)
                if output is not None:
                    output.write(data)
            if tensor_digest.hexdigest() != segment["sha256"]:
                raise ValueError(f"Tensor checksum mismatch: {segment['name']}")
    return digest.hexdigest()


def _manifest_bytes(manifest):
    # Include the manifest itself in actual compact_bytes; its decimal length stabilizes quickly.
    payload_bytes = sum(item["bytes"] for item in manifest["files"].values())
    for _ in range(8):
        data = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        compact_bytes = payload_bytes + len(data)
        if manifest.get("compact_bytes") == compact_bytes:
            return data
        manifest["compact_bytes"] = compact_bytes
    raise ValueError("Could not determine compact bundle size")


def pack_checkpoint(checkpoint_dir, base_dir, destination):
    """Return a verified manifest after saving an exact compact checkpoint bundle.

    Input is a completed ``pretrained_model`` directory, not a training run root.
    Non-prompt changes are preserved as well and may make the bundle large.
    """
    checkpoint, base_root = _directory(checkpoint_dir), _directory(base_dir)
    source_path, base_path = checkpoint / MODEL, base_root / MODEL
    files_before, directories = _inventory(checkpoint)
    for name in (MODEL, *REQUIRED_SIDECARS):
        if name not in files_before:
            raise ValueError(f"Missing required checkpoint file: {checkpoint / name}")
    source_before, base_before = _fingerprint(source_path), _fingerprint(base_path)
    source, base = _model_header(source_path), _model_header(base_path)
    base_sha = _sha256(base_path)
    segments, source_sha, delta_bytes = _build_plan(source_path, source, base_path, base)
    _unchanged(source_path, source_before)
    _unchanged(base_path, base_before)
    sidecars = {name: fp for name, fp in files_before.items() if name != MODEL}
    with _output_directory(destination, (checkpoint, base_root)) as staging:
        # Include a conservative allowance for the JSON manifest; no full model copy is needed.
        _check_space(staging, delta_bytes + len(source["prefix"]) + sum(x[2] for x in sidecars.values())
                     + max(1024 * 1024, len(segments) * 512))
        with (staging / HEADER).open("xb") as stream:
            stream.write(source["prefix"])
            stream.flush()
            os.fsync(stream.fileno())
        delta_digest = hashlib.sha256()
        with source_path.open("rb") as incoming, (staging / DELTA).open("xb") as outgoing:
            for segment in segments:
                if segment["source"] == "delta":
                    start = source["tensors"][segment["name"]]["data_offsets"][0]
                    for data in _region(incoming, len(source["prefix"]) + start, segment["bytes"]):
                        outgoing.write(data)
                        delta_digest.update(data)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        saved_files = {
            HEADER: {"bytes": len(source["prefix"]), "sha256": hashlib.sha256(source["prefix"]).hexdigest()},
            DELTA: {"bytes": delta_bytes, "sha256": delta_digest.hexdigest()},
        }
        for directory in directories:
            (staging / "files" / _relative(directory)).mkdir(parents=True, exist_ok=True)
        for name in sorted(sidecars):
            saved_files[f"files/{name}"] = _copy_hashed(checkpoint / name, staging / "files" / name)
        if _reconstruct(source, segments, base_path, staging / DELTA) != source_sha:
            raise ValueError("Reconstructed model checksum differs from the source checkpoint")
        if _inventory(checkpoint)[0] != files_before:
            raise ValueError("Checkpoint contents changed while packing")
        _unchanged(base_path, base_before)
        manifest = {
            "format": FORMAT,
            "base_model_sha256": base_sha,
            "base_model_bytes": base["size"],
            "source_model_sha256": source_sha,
            "source_model_bytes": source["size"],
            "source_checkpoint_bytes": sum(fp[2] for fp in files_before.values()),
            "stored_tensor_bytes": delta_bytes,
            "reused_tensor_bytes": sum(x["bytes"] for x in segments if x["source"] == "base"),
            "stored_tensor_keys": [x["name"] for x in segments if x["source"] == "delta"],
            "reused_tensor_keys": [x["name"] for x in segments if x["source"] == "base"],
            "base_only_tensor_keys": sorted(set(base["tensors"]) - {
                alias for name in source["tensors"] for alias in _base_key_candidates(name)
            }),
            "segments": segments,
            "directories": directories,
            "files": saved_files,
            "reconstruction_verified": True,
            "note": "Requires the exact base model hash; materialize before standard Piper deployment.",
        }
        with (staging / MANIFEST).open("xb") as stream:
            stream.write(_manifest_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
    return manifest


def _validate_bundle(bundle, base_root):
    before, _ = _inventory(bundle)
    manifest = _json_loads(_regular_file(bundle / MANIFEST).read_bytes())
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint bundle format")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not {HEADER, DELTA}.issubset(entries):
        raise ValueError("Bundle file inventory is missing")
    for name, item in entries.items():
        relative = _relative(name)
        if name not in (HEADER, DELTA) and (relative.parts[0] != "files" or len(relative.parts) < 2):
            raise ValueError(f"Unexpected bundle file: {name}")
        if name == f"files/{MODEL}":
            raise ValueError("Bundle sidecars cannot replace the reconstructed model")
        if not isinstance(item, dict) or type(item.get("bytes")) is not int or item["bytes"] < 0:
            raise ValueError(f"Invalid bundle file metadata: {name}")
        path = _regular_file(bundle / relative)
        if path.stat().st_size != item["bytes"] or _sha256(path) != item.get("sha256"):
            raise ValueError(f"Bundle file checksum mismatch: {name}")
    if set(before) != set(entries) | {MANIFEST}:
        raise ValueError("Unexpected or missing files in checkpoint bundle")
    if any(f"files/{name}" not in entries for name in REQUIRED_SIDECARS):
        raise ValueError("Bundle is missing checkpoint config/processor artifacts")
    directories = manifest.get("directories", [])
    if not isinstance(directories, list):
        raise ValueError("Invalid checkpoint directory inventory")
    for name in directories:
        _relative(name)
    base_path = _regular_file(base_root / MODEL)
    base_before = _fingerprint(base_path)
    if (
        base_before[2] != manifest.get("base_model_bytes")
        or _sha256(base_path) != manifest.get("base_model_sha256")
    ):
        raise ValueError("Base model checksum mismatch; use the exact original shared base")
    base = _model_header(base_path)
    source = _parse_header((bundle / HEADER).read_bytes(), manifest.get("source_model_bytes"))
    segments = manifest.get("segments")
    if not isinstance(segments, list) or len(segments) != len(source["tensors"]):
        raise ValueError("Invalid tensor segment inventory")
    delta_cursor = 0
    for name, segment in zip(_ordered_tensors(source), segments, strict=True):
        tensor = source["tensors"][name]
        size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        if (
            not isinstance(segment, dict) or segment.get("name") != name
            or type(segment.get("bytes")) is not int or segment["bytes"] != size
            or type(segment.get("offset")) is not int or segment["offset"] < 0
            or not isinstance(segment.get("sha256"), str) or len(segment["sha256"]) != 64
        ):
            raise ValueError(f"Invalid tensor segment: {name}")
        if segment.get("source") == "base":
            base_key = segment.get("base_key", name)
            if base_key not in _base_key_candidates(name):
                raise ValueError(f"Unsupported base tensor alias: {name}")
            candidate = base["tensors"].get(base_key)
            if (candidate is None or any(candidate[key] != tensor[key] for key in ("dtype", "shape"))
                    or segment["offset"] != len(base["prefix"]) + candidate["data_offsets"][0]):
                raise ValueError(f"Base tensor does not match bundle: {name}")
        elif segment.get("source") == "delta":
            if segment["offset"] != delta_cursor or segment.get("base_key") is not None:
                raise ValueError(f"Invalid delta tensor offset: {name}")
            delta_cursor += size
        else:
            raise ValueError(f"Invalid tensor source: {name}")
    if delta_cursor != entries[DELTA]["bytes"]:
        raise ValueError("Unaccounted delta tensor bytes")
    _unchanged(base_path, base_before)
    if _inventory(bundle)[0] != before:
        raise ValueError("Bundle contents changed while verifying")
    return manifest, source, segments, base_before, before


def verify_checkpoint(bundle_dir, base_dir):
    """Verify all bundle files and the exact reconstructed model without writing it."""
    bundle, base_root = _directory(bundle_dir), _directory(base_dir)
    manifest, source, segments, base_before, bundle_before = _validate_bundle(bundle, base_root)
    if _reconstruct(source, segments, base_root / MODEL, bundle / DELTA) != manifest["source_model_sha256"]:
        raise ValueError("Reconstructed model checksum mismatch")
    _unchanged(base_root / MODEL, base_before)
    if _inventory(bundle)[0] != bundle_before:
        raise ValueError("Bundle contents changed while reconstructing")
    return manifest


def materialize_checkpoint(bundle_dir, base_dir, destination):
    """Rebuild the exact saved model/config/processors into a new deployment directory."""
    bundle, base_root = _directory(bundle_dir), _directory(base_dir)
    manifest, source, segments, base_before, bundle_before = _validate_bundle(bundle, base_root)
    needed = source["size"] + sum(item["bytes"] for name, item in manifest["files"].items()
                                  if name.startswith("files/"))
    with _output_directory(destination, (bundle, base_root)) as staging:
        _check_space(staging, needed)
        for directory in manifest.get("directories", []):
            (staging / _relative(directory)).mkdir(parents=True, exist_ok=True)
        for name, item in manifest["files"].items():
            if name.startswith("files/"):
                copied = _copy_hashed(bundle / name, staging / _relative(name.removeprefix("files/")))
                if copied != item:
                    raise ValueError(f"Copied processor/config checksum mismatch: {name}")
        with (staging / MODEL).open("xb") as output:
            checksum = _reconstruct(source, segments, base_root / MODEL, bundle / DELTA, output)
            output.flush()
            os.fsync(output.fileno())
        if checksum != manifest["source_model_sha256"]:
            raise ValueError("Reconstructed model checksum mismatch")
        _unchanged(base_root / MODEL, base_before)
        if _inventory(bundle)[0] != bundle_before:
            raise ValueError("Bundle contents changed while reconstructing")
    return Path(destination).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Create and verify a compact checkpoint; keep the original")
    pack.add_argument("--checkpoint", required=True, type=Path)
    pack.add_argument("--base", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    materialize = commands.add_parser("materialize", help="Reconstruct a normal Piper pretrained_model")
    materialize.add_argument("--bundle", required=True, type=Path)
    materialize.add_argument("--base", required=True, type=Path)
    materialize.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser(
        "verify", help="Verify exact reconstruction without allocating a full model file"
    )
    verify.add_argument("--bundle", required=True, type=Path)
    verify.add_argument("--base", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "pack":
            result = pack_checkpoint(args.checkpoint, args.base, args.output)
            print(json.dumps({key: result[key] for key in (
                "compact_bytes", "source_checkpoint_bytes", "stored_tensor_bytes",
                "base_model_sha256", "source_model_sha256", "reconstruction_verified",
            )}, indent=2))
        elif args.command == "materialize":
            print(materialize_checkpoint(args.bundle, args.base, args.output))
        else:
            result = verify_checkpoint(args.bundle, args.base)
            print(f"Verified exact model: {result['source_model_sha256']}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Checkpoint operation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
