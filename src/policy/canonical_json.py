"""The repository's single stdlib-only canonical JSON implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CanonicalJSONError(ValueError):
    """Raised when input cannot satisfy the canonical JSON contract."""


def canonical_bytes(document: Any) -> bytes:
    """Return RFC-8259-compatible canonical bytes without a trailing newline."""

    try:
        text = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalJSONError(str(exc)) from exc
    return text.encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: str | os.PathLike[str]) -> Any:
    """Load UTF-8 JSON while rejecting CRLF, duplicate keys, and non-finite values."""

    raw = Path(path).read_bytes()
    if b"\r" in raw:
        raise CanonicalJSONError("carriage returns are not canonical JSON input")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CanonicalJSONError("input is not UTF-8") from exc

    def reject_constant(value: str) -> Any:
        raise CanonicalJSONError(f"non-finite JSON number: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise CanonicalJSONError(str(exc)) from exc


def identity_for(document: Mapping[str, Any], identity_field: str) -> str:
    """Hash a shallow copy after removing exactly ``identity_field``."""

    if not identity_field:
        raise CanonicalJSONError("identity field must be non-empty")
    payload = dict(document)
    payload.pop(identity_field, None)
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def materialize_identity(
    document: Mapping[str, Any], identity_field: str
) -> dict[str, Any]:
    """Return a complete document with its non-recursive identity populated."""

    result = dict(document)
    result.pop(identity_field, None)
    result[identity_field] = identity_for(result, identity_field)
    return result


def validate_identity(document: Mapping[str, Any], identity_field: str) -> bool:
    """Return whether the identity is present and exactly matches the document."""

    actual = document.get(identity_field)
    return isinstance(actual, str) and actual == identity_for(document, identity_field)


def publish_canonical_no_clobber(output: str | os.PathLike[str], document: Any) -> None:
    """Atomically publish canonical JSON plus LF without replacing an existing path."""

    target = Path(output)
    target.parent.mkdir(parents=False, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.partial."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(document))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_to_caller_temp(output: Path, document: Any) -> None:
    """Write a caller-owned temporary path used before a no-clobber hard link."""

    if output.is_symlink():
        raise CanonicalJSONError("output temporary path must not be a symlink")
    with output.open("wb") as handle:
        handle.write(canonical_bytes(document))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("materialize-id", "publish-id"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--input", required=True, type=Path)
        subparser.add_argument("--identity-field", required=True)
        subparser.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser("validate-id")
    validate.add_argument("--input", required=True, type=Path)
    validate.add_argument("--identity-field", required=True)
    validate.add_argument("--require-schema")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = load_json_strict(args.input)
        if not isinstance(document, dict):
            raise CanonicalJSONError("identity documents must be JSON objects")

        if args.command == "validate-id":
            if (
                args.require_schema is not None
                and document.get("schema_version") != args.require_schema
            ):
                return 2
            return 0 if validate_identity(document, args.identity_field) else 2

        materialized = materialize_identity(document, args.identity_field)
        if args.command == "materialize-id":
            _materialize_to_caller_temp(args.output, materialized)
        else:
            publish_canonical_no_clobber(args.output, materialized)
        return 0
    except (CanonicalJSONError, FileExistsError, OSError) as exc:
        print(f"canonical-json: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
